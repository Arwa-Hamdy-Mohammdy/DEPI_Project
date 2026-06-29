"""
main.py
=======
Async drone control loop — PPO inference edition.

This replaces the legacy DQN integration. The agent loaded here is an
SB3 PPO policy trained inside ``DroneCityEnv`` (intelligence/DroneCityEnv.py),
which is the single source of truth for the observation/action contract:

    observation (spaces.Dict):
        "kinematics"      (6,)        [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]  (ENU)
        "waypoint_vector" (3,)        target_pos - current_pos                    (ENU)
        "vision"          (64, 64, 1) depth image, Building class masked to 0.0,
                                       normalised to [0, 1]
    action (spaces.Box, shape (3,)):
        [vx, vy, vz] continuous velocity command in m/s, ENU, clipped to
        [-MAX_VEL, MAX_VEL]

Everything in this file exists to assemble that exact observation, every
control step, from live AirSim/vision data, and to apply the resulting
action with the correct ENU→NED sign convention. See the inline notes
flagged "PPO PARITY" for the specific points where a mismatch with
DroneCityEnv would silently degrade flight behaviour.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
from enum import Enum
from pathlib import Path
from typing import Dict, List

import numpy as np

from controllers.drone_controller import DroneController
from intelligence.rl_agent import RLAgent
from services.vision_service import VisionService
from services.flight_narrator import FlightNarrator
from utils.logger import setup_logger


class MissionType(Enum):
    STEPS = "steps"
    GOAL = "goal"
    CITY_TOUR = "city_tour"


# ═══════════════════════════════════════════════════════════════════════════════
# PPO Inference Constants
# These MUST mirror the corresponding constants in DroneCityEnv.py exactly.
# Any drift here silently breaks train/inference parity.
# ═══════════════════════════════════════════════════════════════════════════════
MAX_VEL: float = 5.0             # m/s   — DroneCityEnv.MAX_VEL
ACTION_DURATION_S: float = 0.1   # s     — DroneCityEnv.STEP_DURATION_S
WAYPOINT_RADIUS_M: float = 2.0   # m     — DroneCityEnv.WAYPOINT_RADIUS_M
VISION_SHAPE: tuple[int, int, int] = (64, 64, 1)  # DroneCityEnv vision obs shape


# ═══════════════════════════════════════════════════════════════════════════════
# Main Control Loop
# ═══════════════════════════════════════════════════════════════════════════════
async def run_control_loop(
    mission_type: MissionType = MissionType.STEPS,
    max_steps: int = 100,
    target_distance_m: float = 100.0,
    step_delay_s: float = 0.0,
    goal_delay_s: float = 0.0,
) -> None:
    project_root = Path(__file__).resolve().parent
    logger = setup_logger(log_dir=project_root / "data" / "logs")
    logger.info("🚀 Starting ASYNC drone control loop (PPO inference). Mode: %s", mission_type.value)

    vision = VisionService(project_root=project_root, logger=logger,model_filename="yolo11n.pt")
    controller = DroneController(logger=logger)
    narrator = FlightNarrator(logger=logger)

    agent = RLAgent(
        logger=logger,
        model_path=project_root / "models" / "ppo_drone_final.zip",
    )
    agent.set_eval_mode()  # deterministic PPO actions for flight

    # ── Pre-flight dependency check ─────────────────────────────────────────
    # PPO PARITY: fail here, before arming/takeoff, rather than mid-flight.
    # The policy was trained on AirSim's filtered kinematics_estimated
    # velocity, NOT a finite-difference approximation of pose samples — the
    # two are not interchangeable. See the `get_kinematics()` note below.
    if not hasattr(controller, "get_kinematics"):
        raise RuntimeError(
            "DroneController has no get_kinematics() method. The PPO policy "
            "needs AirSim's filtered velocity estimate (matching "
            "DroneCityEnv._get_obs()), not a finite-differenced pose delta. "
            "Add get_kinematics() to controllers/drone_controller.py before "
            "flying — see the accompanying snippet."
        )
    if not hasattr(vision, "get_ppo_vision_obs"):
        raise RuntimeError("VisionService has no get_ppo_vision_obs() method.")

    try:
        await controller.connect()
        await controller.takeoff()
        await asyncio.sleep(2.0)  # let the drone settle in the air

        if mission_type == MissionType.STEPS:
            await _run_step_mission(controller, vision, agent, narrator, logger, max_steps, step_delay_s)
        elif mission_type == MissionType.GOAL:
            await _run_goal_mission(controller, vision, agent, narrator, logger, target_distance_m, goal_delay_s)
        elif mission_type == MissionType.CITY_TOUR:
            # PPO PARITY: z is ENU (positive = up), matching
            # DroneCityEnv's observation convention — NOT AirSim's raw NED
            # (negative = up). The old DQN waypoints used -20.0; that value
            # would now be read as "20 m *below* ground".
            waypoints = [
                {"x": 50.0, "y": 0.0, "z": 20.0},
                {"x": 50.0, "y": 50.0, "z": 20.0},
                {"x": -50.0, "y": 50.0, "z": 20.0},
            ]
            await _run_city_tour_mission(controller, vision, agent, narrator, logger, waypoints, goal_delay_s)

    except KeyboardInterrupt:
        logger.warning("⚠️ Control loop interrupted by operator (KeyboardInterrupt).")
    except asyncio.CancelledError:
        logger.warning("⚠️ Mission task cancelled.")
    except Exception as exc:
        logger.exception("❌ CRITICAL error in control loop: %s", exc)
    finally:
        logger.info("🛡️ Initiating safety shutdown.")
        try:
            await controller.hover()   # stop in place to absorb momentum
            await asyncio.sleep(1.0)   # let physics settle
            await controller.land()
        except Exception as e:
            logger.error("Error during safety landing: %s", e)

        await controller.disconnect()
        vision.shutdown()
        logger.info("🏁 Control loop terminated safely.")


# ═══════════════════════════════════════════════════════════════════════════════
# Mission Modes
# ═══════════════════════════════════════════════════════════════════════════════
async def _run_step_mission(
    controller: DroneController,
    vision: VisionService,
    agent: RLAgent,
    narrator: FlightNarrator,
    logger: logging.Logger,
    max_steps: int,
    delay: float,
) -> None:
    dummy_target = {"x": 0.0, "y": 0.0, "z": 0.0}  # placeholder goal, preserves obs shape

    for step in range(max_steps):
        kinematics, vision_obs, raw_frame = await _gather_observation_inputs(controller, vision)

        obs = _build_consistent_state(kinematics, dummy_target, vision_obs)
        action = await agent.select_action(obs)

        # await _execute_movement(controller, action)
        # logger.info("Step %s completed. action=[%.2f, %.2f, %.2f]", step, action[0], action[1], action[2])

        # # Telemetry only from here down — NOT part of the PPO observation.
        # detections = await vision.detect_objects(raw_frame)
        # narrator.narrate(
        #     action=action.tolist(),
        #     pose=_pos_to_dict(kinematics["position"]),
        #     velocity=_pos_to_dict(kinematics["velocity"]),
        #     target=dummy_target,
        #     detections=detections,
        #     step=step,
        # )
        # asyncio.create_task(vision.save_detected_frame(raw_frame, detections))

        # await asyncio.sleep(delay)

async def _execute_movement(
controller: DroneController,
action: np.ndarray,
duration: float = ACTION_DURATION_S,
) -> None:
    clipped = np.clip(action, -MAX_VEL, MAX_VEL)
    vx, vy, vz = float(clipped[0]), float(clipped[1]), float(clipped[2])

    # التعديل الهندسي: 
    # نشغل أمر الحركة في الخلفية (Background Task) لمدة 5 ثواني متواصلة.
    # الدرون هيفضل طاير بـ Momentum ناعم جداً، وأول ما اللوب تخلص معالجة الصورة 
    # هتبعت أمر جديد لـ AirSim يلغي القديم فوراً ويصحح المسار بدقة.
    asyncio.create_task(controller.move(vx, vy, -vz, 5.0))

async def _run_goal_mission(
    controller: DroneController,
    vision: VisionService,
    agent: RLAgent,
    narrator: FlightNarrator,
    logger: logging.Logger,
    target_distance: float,
    delay: float,
) -> None:
    start_kinematics = await controller.get_kinematics()
    start_pos = start_kinematics["position"]  # (3,) float32, ENU

    # Project a target straight ahead on the x-axis at the current altitude,
    # exactly mirroring the old behaviour but in the ENU frame.
    estimated_target = {
        "x": float(start_pos[0]) + target_distance,
        "y": float(start_pos[1]),
        "z": float(start_pos[2]),
    }

    reached_target = False
    while not reached_target:
        kinematics, vision_obs, raw_frame = await _gather_observation_inputs(controller, vision)
        pos = kinematics["position"]
        distance_covered = _calculate_distance(start_pos, pos)

        if distance_covered >= target_distance:
            logger.info("\n✅ Goal Reached! %.2f meters covered.", distance_covered)
            reached_target = True
            break

        obs = _build_consistent_state(kinematics, estimated_target, vision_obs)
        action = await agent.select_action(obs)
        await _execute_movement(controller, action)

        detections = await vision.detect_objects(raw_frame)
        narrator.narrate(
            action=action.tolist(),
            pose=_pos_to_dict(pos),
            velocity=_pos_to_dict(kinematics["velocity"]),
            target=estimated_target,
            detections=detections,
        )
        print(f"Distance covered: {distance_covered:.2f} / {target_distance}m", end="\r")
        asyncio.create_task(vision.save_detected_frame(raw_frame, detections))

        await asyncio.sleep(delay)


async def _run_city_tour_mission(
    controller: DroneController,
    vision: VisionService,
    agent: RLAgent,
    narrator: FlightNarrator,
    logger: logging.Logger,
    waypoints: List[Dict[str, float]],
    delay: float,
) -> None:
    logger.info("🏙️ Starting City Tour. Total waypoints: %s", len(waypoints))

    for index, target in enumerate(waypoints):
        logger.info("\n--- Navigating to Waypoint %s: %s ---", index + 1, target)
        await _run_single_waypoint(controller, vision, agent, narrator, logger, target, delay)
        logger.info("📍 Waypoint %s reached successfully!", index + 1)

    logger.info("🎉 City Tour Complete! All waypoints visited.")


async def _run_single_waypoint(
    controller: DroneController,
    vision: VisionService,
    agent: RLAgent,
    narrator: FlightNarrator,
    logger: logging.Logger,
    target_pose: Dict[str, float],
    delay: float,
) -> None:
    target_arr = np.array(
        [target_pose["x"], target_pose["y"], target_pose.get("z", 0.0)], dtype=np.float32
    )
    reached_target = False

    while not reached_target:
        kinematics, vision_obs, raw_frame = await _gather_observation_inputs(controller, vision)
        pos = kinematics["position"]
        distance_to_target = _calculate_distance(pos, target_arr)

        if distance_to_target <= WAYPOINT_RADIUS_M:
            reached_target = True
            break

        obs = _build_consistent_state(kinematics, target_pose, vision_obs)
        action = await agent.select_action(obs)
        await _execute_movement(controller, action)

        detections = await vision.detect_objects(raw_frame)
        narrator.narrate(
            action=action.tolist(),
            pose=_pos_to_dict(pos),
            velocity=_pos_to_dict(kinematics["velocity"]),
            target=target_pose,
            detections=detections,
        )
        asyncio.create_task(vision.save_detected_frame(raw_frame, detections))
        print(
            f"IN FLIGHT | Target Distance: [ {distance_to_target:>6.2f}m ] | "
            f"Action: [{action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f}]",
            end="\r",
        )

        await asyncio.sleep(delay)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════
async def _gather_observation_inputs(
    controller: DroneController, vision: VisionService
):
    """
    Fetches everything one control step needs in parallel:
      - kinematics (PPO observation input)
      - PPO depth/segmentation vision obs (PPO observation input)
      - RGB frame (telemetry/narration only — NOT fed to the policy)

    Note: `vision.get_ppo_vision_obs()` and `vision.get_frame()` both go
    through VisionService's single dedicated AirSim RPC thread (AirSim's
    client is not safe for concurrent calls), so they execute sequentially
    even under `asyncio.gather`. `controller.get_kinematics()` runs on its
    own channel and overlaps with both. If per-step latency becomes a
    bottleneck, the next optimisation is fusing Scene+Depth+Segmentation
    into a single `simGetImages()` call instead of two.
    """
    return await asyncio.gather(
        controller.get_kinematics(),
        vision.get_ppo_vision_obs(),
        vision.get_frame(),
    )


async def _execute_movement(
    controller: DroneController,
    action: np.ndarray,
    duration: float = ACTION_DURATION_S,
) -> None:
    """
    Sends the PPO action directly as a velocity command. Mirrors
    DroneCityEnv.step() exactly:

        action = np.clip(action, -MAX_VEL, MAX_VEL)
        client.moveByVelocityAsync(vx, vy, vz=-action_vz, duration=STEP_DURATION_S,
                                    drivetrain=MaxDegreeOfFreedom,
                                    yaw_mode=YawMode(is_rate=False, yaw_or_rate=0))

    Only vz is sign-flipped (ENU → AirSim NED); vx/vy pass straight through.

    PPO PARITY: `controller.move()` must issue the move with
    `drivetrain=DrivetrainType.MaxDegreeOfFreedom` and a fixed
    `yaw_mode=YawMode(is_rate=False, yaw_or_rate=0)`, same as training. If it
    instead defaults to "face direction of travel" (a common ForwardOnly
    default), the drone's actual heading — and therefore what the depth
    camera sees — will diverge from what the policy was trained on, even
    though the velocity vector itself is correct. Verify this in
    drone_controller.py before flying.
    """
    clipped = np.clip(action, -MAX_VEL, MAX_VEL)
    vx, vy, vz = float(clipped[0]), float(clipped[1]), float(clipped[2])
    await controller.move(vx, vy, -vz, duration)


def _build_consistent_state(
    kinematics: Dict[str, np.ndarray],
    target_pose: Dict[str, float],
    vision_obs: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Builds the SB3 PPO observation dict, matching
    DroneCityEnv.observation_space exactly:

        "kinematics":      (6,)        [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]  ENU
        "waypoint_vector": (3,)        target_pos - current_pos                    ENU
        "vision":          (64, 64, 1) depth image, buildings masked, in [0, 1]

    Unlike the old DQN `_build_consistent_state`, this needs no
    `previous_pose` / `last_time` bookkeeping at all: velocity now comes
    straight from AirSim's kinematics estimate (via
    `controller.get_kinematics()`), not a finite difference of pose
    samples — removing an entire class of dt/divide-by-zero bugs from the
    mission loops above.
    """
    pos = kinematics["position"]  # (3,) float32, ENU
    vel = kinematics["velocity"]  # (3,) float32, ENU

    kinematics_obs = np.concatenate([pos, vel]).astype(np.float32)

    waypoint_vector = np.array(
        [
            target_pose["x"] - float(pos[0]),
            target_pose["y"] - float(pos[1]),
            target_pose.get("z", 0.0) - float(pos[2]),
        ],
        dtype=np.float32,
    )

    vision_obs = vision_obs.astype(np.float32)
    assert vision_obs.shape == VISION_SHAPE, (
        f"vision obs shape {vision_obs.shape} != expected {VISION_SHAPE} — "
        "check VisionService.get_ppo_vision_obs()."
    )

    return {
        "kinematics": kinematics_obs,
        "waypoint_vector": waypoint_vector,
        "vision": vision_obs,
    }


def _pos_to_dict(vec: np.ndarray) -> Dict[str, float]:
    """Small adapter so FlightNarrator keeps receiving {'x','y','z'} dicts."""
    return {"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2])}


def _calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """3-D Euclidean distance between two ENU position vectors."""
    return float(np.linalg.norm(np.asarray(p1, dtype=np.float32) - np.asarray(p2, dtype=np.float32)))


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════
async def main():
    print("----------------- MODES ------------------")
    print("1- STEPS , 2- GOAL , 3- CITY-TOUR")

    valid = False
    _mission_type = MissionType.STEPS
    try:
        while not valid:
            x = input("ENTER NUM OF MODE : ")
            if x == "1":
                _mission_type = MissionType.STEPS
                valid = True
            elif x == "2":
                _mission_type = MissionType.GOAL
                valid = True
            elif x == "3":
                _mission_type = MissionType.CITY_TOUR
                valid = True
            else:
                print("Invalid Number")
    except (ValueError, TypeError):
        print("[!] Error: Please enter a numeric value only.")

    await run_control_loop(mission_type=_mission_type)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    asyncio.run(main())
