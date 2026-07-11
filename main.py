from __future__ import annotations

import asyncio
import math
import time
import logging
import multiprocessing as mp
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from stable_baselines3 import PPO

from controllers.drone_controller import DroneController
from intelligence.PathPlanner import PathPlanner
from intelligence.rl_agent import RLAgent
from services.vision_service import VisionService
from services.flight_narrator import FlightNarrator
from utils.logger import setup_logger

class MissionType(Enum):
    STEPS = "steps"
    GOAL = "goal"
    CITY_TOUR = "city_tour"

# ═══════════════════════════════════════════════════════════════════════════════
# Main Control Loop
# ═══════════════════════════════════════════════════════════════════════════════
async def run_control_loop(
    mission_type: MissionType = MissionType.STEPS,
    max_steps: int = 100,
    target_distance_m: float = 100.0,
    step_delay_s: float = 0.1,
    goal_delay_s: float = 0.05,
) -> None:
    project_root = Path(__file__).resolve().parent
    logger = setup_logger(log_dir=project_root / "data" / "logs")
    logger.info("🚀 Starting ASYNC drone control loop. Mode: %s", mission_type.value)

    vision = VisionService(project_root=project_root, logger=logger, device='cuda', min_confidence=0.35)
    controller = DroneController(logger=logger)
    dummy_grid = np.zeros((10, 10, 10), dtype=np.uint8)
    planner = PathPlanner(voxel_grid=dummy_grid)
    narrator = FlightNarrator(logger=logger)
    
    # [1] Load the trained PPO model
    model_path = project_root / "models" / "best_model" / "best_model.zip"
    if not model_path.exists():
        model_path = project_root / "models" / "ppo_drone_final.zip"
        if not model_path.exists():
            logger.warning("Could not find best_model.zip or ppo_drone_final.zip. Attempting to load from checkpoints...")
            # Fallback to the latest checkpoint if possible, or let it fail naturally
            
    # Provide custom objects to fix FloatSchedule missing attribute exception
    # during deserialization of lr_schedule and clip_range
    custom_objects = {
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    agent = PPO.load(model_path, custom_objects=custom_objects)

    try:
        await controller.connect()
        await controller.takeoff()
        await asyncio.sleep(2.0) # انتظار استقرار الدرون في الهواء

        if mission_type == MissionType.STEPS:
            await _run_step_mission(controller, vision, planner, agent, narrator, logger, max_steps, step_delay_s)
        elif mission_type == MissionType.GOAL:
            await _run_goal_mission(controller, vision, planner, agent, narrator, logger, target_distance_m, goal_delay_s)
        elif mission_type == MissionType.CITY_TOUR:
            waypoints = [
                {"x": 50.0, "y": 0.0, "z": -20.0},
                {"x": 50.0, "y": 50.0, "z": -20.0},
                {"x": -50.0, "y": 50.0, "z": -20.0},
            ]
            await _run_city_tour_mission(controller, vision, planner, agent, narrator, logger, waypoints, goal_delay_s)

    except KeyboardInterrupt:
        logger.warning("⚠️ Control loop interrupted by operator (KeyboardInterrupt).")
    except asyncio.CancelledError:
        logger.warning("⚠️ Mission task cancelled.")
    except Exception as exc:
        logger.exception("❌ CRITICAL error in control loop: %s", exc)
    finally:
        # [2] نظام الهبوط الآمن (Fail-Safe)
        logger.info("🛡️ Initiating safety shutdown.")
        try:
            await controller.hover()  # إيقاف المحركات في مكانها لامتصاص الزخم
            await asyncio.sleep(1.0)  # انتظار استقرار الفيزياء
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
    controller: DroneController, vision: VisionService, planner: PathPlanner,
    agent: RLAgent, narrator: FlightNarrator, logger: logging.Logger, max_steps: int, delay: float,
) -> None:
    
    # هدف وهمي للحفاظ على شكل الـ State
    dummy_target = {"x": 0.0, "y": 0.0, "z": 0.0} 
    previous_pose = await controller.get_pose()
    last_time = time.time()

    for step in range(max_steps):
        frames, current_pose, current_yaw = await asyncio.gather(
            vision.get_frames(),
            controller.get_pose(),
            controller.get_yaw(),
        )

        rgb_frame = frames.get("rgb", np.array([]))
        seg_frame = frames.get("seg", np.array([]))
        detections = await vision.get_fused_detections(rgb_frame, seg_frame)
        current_time = time.time()
        
        # [3] توحيد بناء مساحة الحالة (State) وحساب السرعة
        state, velocity = _build_consistent_state(current_pose, previous_pose, dummy_target, frames, current_time, last_time, current_yaw)

        action, _ = agent.predict(state, deterministic=True)
        await controller.move(float(action[0]), float(action[1]), -float(action[2]), 1.0)
        
        logger.info("Step %s completed. action=%s", step, action)

        narrator.narrate(
            action=action, pose=current_pose, velocity=velocity,
            target=dummy_target, detections=detections, step=step,
        )

        asyncio.create_task(vision.save_detected_frame(rgb_frame, detections))
        
        previous_pose = current_pose
        last_time = current_time
        await asyncio.sleep(delay)

async def _run_goal_mission(
    controller: DroneController, vision: VisionService, planner: PathPlanner,
    agent: RLAgent, narrator: FlightNarrator, logger: logging.Logger, target_distance: float, delay: float,
) -> None:
    
    start_pose, start_yaw = await asyncio.gather(
        controller.get_pose(),
        controller.get_yaw()
    )
    
    # 1. Dynamic 3D Target Formulation
    # Calculate exact 3D coordinates based on initial position and heading (Yaw)
    yaw_rad = math.radians(start_yaw)
    exact_target = {
        "x": start_pose["x"] + target_distance * math.cos(yaw_rad),
        "y": start_pose["y"] + target_distance * math.sin(yaw_rad),
        "z": start_pose.get("z", -10.0) # Maintain current start altitude
    }
    
    logger.info("Goal Mission initialized. Target calculated at: X=%.2f, Y=%.2f, Z=%.2f", exact_target['x'], exact_target['y'], exact_target['z'])

    previous_pose = start_pose
    last_time = time.time()
    
    # 2. Termination Conditions & Fail-Safes
    ACCEPTANCE_RADIUS = 2.0
    MAX_STEPS = int(600.0 / delay)  # Max 10 minutes timeout fail-safe
    step_count = 0
    reached_target = False
    
    # State machine for persistent lateral evasion
    evasion_direction = 1.0
    is_evading = False

    while not reached_target and step_count < MAX_STEPS:
        frames, current_pose, current_yaw = await asyncio.gather(
            vision.get_frames(),
            controller.get_pose(),
            controller.get_yaw(),
        )
        
        # Calculate real-time 3D Euclidean distance to target
        distance_to_target = _calculate_distance(current_pose, exact_target)
        
        # Target-Centric Termination check
        if distance_to_target <= ACCEPTANCE_RADIUS:
            logger.info("\n✅ Goal Reached! Distance to target is within acceptance radius (%.2fm).", distance_to_target)
            reached_target = True
            await controller.hover()
            break
            
        rgb_frame = frames.get("rgb", np.array([]))
        seg_frame = frames.get("seg", np.array([]))
        detections = await vision.get_fused_detections(rgb_frame, seg_frame)
        current_time = time.time()

        # 3. State Observation Consistency
        state, velocity = _build_consistent_state(current_pose, previous_pose, exact_target, frames, current_time, last_time, current_yaw)
        
        # We rely exclusively on the depth sensor for safety overrides.
        dist_val = float(np.ravel(state["distance_sensor"])[0])
        
        emergency_state = False
        final_vx, final_vy, final_vz = 0.0, 0.0, 0.0
        
        # 1. Evasive Ascent & Wall-Sliding (Imminent Collision < 2.4 meters)
        if dist_val < 0.08:
            if not is_evading:
                # Determine lateral direction based on PPO's underlying preference
                action, _ = agent.predict(state, deterministic=True)
                evasion_direction = 1.0 if float(action[1]) >= 0 else -1.0
                is_evading = True
                
            logger.warning("🚨 PANIC OVERRIDE ENGAGED: Imminent collision! Forcing lateral wall-slide.")
            final_vx, final_vy, final_vz = 0.0, 2.0 * evasion_direction, 0.5  # Slide laterally, slight climb
            emergency_state = True
        else:
            is_evading = False
            # Only query PPO if not in an emergency
            action, _ = agent.predict(state, deterministic=True)
            final_vx, final_vy, final_vz = float(action[0]), float(action[1]), float(action[2])
            
            # 2. Depth Auto-Brake (Obstacle Close < 4.5 meters)
            if dist_val < 0.15:
                if final_vx > 0.2:
                    logger.warning("⚠️ AUTO-BRAKE ENGAGED: Obstacle critically close! Clamping forward speed.")
                    final_vx = 0.2

        # Convert ENU Z-axis (+Up) to AirSim NED Z-axis (-Up) for movement
        await controller.move(final_vx, final_vy, -final_vz, 1.0)
        modified_action = np.array([final_vx, final_vy, final_vz], dtype=np.float32)
        
        # Break Endless Evasive Loop
        if emergency_state:
            if 'panic_start_time' not in locals():
                panic_start_time = current_time
            elif current_time - panic_start_time > 4.0:
                logger.error("🛑 EVASIVE LOOP DETECTED: Drone stuck in panic mode for > 4 seconds. Aborting local path.")
                await controller.hover()
                break  # Exit mission
        else:
            if 'panic_start_time' in locals():
                del panic_start_time # Reset panic timer
            # Standard PPO Logging (Mutually Exclusive)
            narrator.narrate(
                action=modified_action, pose=current_pose, velocity=velocity,
                target=exact_target, detections=detections,
            )
        
        print(f"Distance to Target: {distance_to_target:.2f}m", end="\r")

        asyncio.create_task(vision.save_detected_frame(rgb_frame, detections))
        
        previous_pose = current_pose
        last_time = current_time
        step_count += 1
        await asyncio.sleep(delay)
        
    if step_count >= MAX_STEPS:
        logger.warning("\n⚠️ Goal mission timed out after %s steps. Forcing hover.", MAX_STEPS)
        await controller.hover()

async def _run_city_tour_mission(
    controller: DroneController, vision: VisionService, planner: PathPlanner,
    agent: RLAgent, narrator: FlightNarrator, logger: logging.Logger, waypoints: List[Dict[str, float]], delay: float,
) -> None:
    
    logger.info("🏙️ Starting City Tour. Total waypoints: %s", len(waypoints))
    
    for index, target in enumerate(waypoints):
        logger.info("\n--- Navigating to Waypoint %s: %s ---", index + 1, target)
        await _run_single_waypoint(controller, vision, planner, agent, narrator, logger, target, delay)
        logger.info("📍 Waypoint %s reached successfully!", index + 1)

    logger.info("🎉 City Tour Complete! All waypoints visited.")

async def _run_single_waypoint(
    controller: DroneController, vision: VisionService, planner: PathPlanner,
    agent: RLAgent, narrator: FlightNarrator, logger: logging.Logger, target_pose: Dict[str, float], delay: float,
) -> None:
    
    reached_target = False
    acceptance_radius = 2.0
    
    # 1. Initialize the pose and time BEFORE the loop begins
    current_pose = await controller.get_pose()
    previous_pose = current_pose
    last_time = time.time()

    while not reached_target:
        frames, current_pose, current_yaw = await asyncio.gather(
            vision.get_frames(),
            controller.get_pose(),
            controller.get_yaw(),
        )
        
        distance_to_target = _calculate_distance(current_pose, target_pose)

        if distance_to_target <= acceptance_radius:
            reached_target = True
            break

        rgb_frame = frames.get("rgb", np.array([]))
        seg_frame = frames.get("seg", np.array([]))
        detections = await vision.get_fused_detections(rgb_frame, seg_frame)
        current_time = time.time()
        
        state, velocity = _build_consistent_state(
            current_pose, previous_pose, target_pose, frames, current_time, last_time, current_yaw
        )
        
        action, _ = agent.predict(state, deterministic=True)
        await controller.move(float(action[0]), float(action[1]), -float(action[2]), 1.0)

        narrator.narrate(
            action=action, pose=current_pose, velocity=velocity,
            target=target_pose, detections=detections,
        )

        asyncio.create_task(vision.save_detected_frame(rgb_frame, detections))
        print(
            f"IN FLIGHT | Target Distance: [ {distance_to_target:>6.2f}m ] | Action: {action}",
            end="\r",
        )
        

        
        # 3. Update the trackers at the end of the iteration
        previous_pose = current_pose
        last_time = current_time
        
        await asyncio.sleep(delay)



# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════
def _build_consistent_state(
    current_pose: Dict[str, float], 
    previous_pose: Dict[str, float], 
    target_pose: Dict[str, float], 
    frames: Dict[str, np.ndarray],
    current_time: float, 
    last_time: float,
    yaw: float = 0.0,
) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Builds the observation dict exactly as PPO expects from DroneCityEnv."""
    dt = current_time - last_time
    if dt <= 0:
        dt = 0.001
        
    velocity = {
        "x": (current_pose["x"] - previous_pose["x"]) / dt,
        "y": (current_pose["y"] - previous_pose["y"]) / dt,
        "z": -(current_pose.get("z", 0) - previous_pose.get("z", 0)) / dt, # Convert to ENU (+Up)
    }
    
    # 1. Kinematics (Horizontal positions zeroed to prevent overfitting, but altitude pos_z is kept!)
    pos_x, pos_y = 0.0, 0.0
    pos_z = -current_pose.get("z", 0.0)  # ENU z
    vel_x, vel_y, vel_z = velocity["x"], velocity["y"], velocity["z"]
    kinematics = np.array([pos_x, pos_y, pos_z, vel_x, vel_y, vel_z], dtype=np.float32)

    # 2. Waypoint Vector
    # Calculate vector from true current position to target
    actual_pos_x, actual_pos_y, actual_pos_z = current_pose["x"], current_pose["y"], -current_pose.get("z", 0.0)
    tx, ty, tz = target_pose["x"], target_pose["y"], -target_pose.get("z", 0.0)
    waypoint_vector = np.array([tx - actual_pos_x, ty - actual_pos_y, tz - actual_pos_z], dtype=np.float32)

    # 3. Vision
    BUILDING_SEG_ID = 12
    MAX_DEPTH_M = 50.0
    H, W = 64, 64
    
    depth_raw = frames.get("depth", np.zeros((H, W), dtype=np.float32)).copy()
    seg_raw = frames.get("seg", np.zeros((H, W, 3), dtype=np.uint8))
    if depth_raw.size == 0:
        depth_raw = np.zeros((H, W), dtype=np.float32)
        
    # Clean NaNs and Infs from depth map to prevent PPO from outputting NaNs (which causes UE to despawn the drone)
    depth_raw = np.nan_to_num(depth_raw, nan=MAX_DEPTH_M, posinf=MAX_DEPTH_M, neginf=0.0)
        
    # 3.5 Center Distance Sensor (Matches DroneCityEnv implementation)
    ph, pw = depth_raw.shape
    crop_h = max(1, ph // 5)
    crop_w = max(1, pw // 5)
    start_h = (ph - crop_h) // 2
    start_w = (pw - crop_w) // 2
    
    center_crop = depth_raw[start_h : start_h + crop_h, start_w : start_w + crop_w]
    
    # Filter out 0.0 artifacts (AirSim returns 0 for sky/invalid in some configs)
    valid_depths = center_crop[center_crop > 0.1]
    min_depth = float(np.min(valid_depths)) if valid_depths.size > 0 else 30.0
    
    dist_norm_scalar = np.clip(min_depth, 0.0, 30.0) / 30.0
    distance_obs = np.array([dist_norm_scalar], dtype=np.float32)

    def _resize(img, out_h, out_w):
        src_h, src_w = img.shape[:2]
        row_idx = (np.arange(out_h) * src_h / out_h).astype(int)
        col_idx = (np.arange(out_w) * src_w / out_w).astype(int)
        if len(img.shape) == 3:
            return img[np.ix_(row_idx, col_idx, np.arange(img.shape[2]))]
        return img[np.ix_(row_idx, col_idx)]

    if seg_raw.size > 0:
        seg_channel = seg_raw[:, :, 0] if len(seg_raw.shape) == 3 else seg_raw
        if seg_channel.shape[:2] != depth_raw.shape[:2]:
            seg_channel = _resize(seg_channel, depth_raw.shape[0], depth_raw.shape[1])
        # Removed logic that zeroes out depth for buildings, as it blinds the agent

    if depth_raw.shape[:2] != (H, W):
        depth_raw = _resize(depth_raw, H, W)

    depth_norm = np.clip(depth_raw, 0.0, MAX_DEPTH_M) / MAX_DEPTH_M
    vision = depth_norm[:, :, np.newaxis].astype(np.float32)

    state = {
        "kinematics": kinematics,
        "waypoint_vector": waypoint_vector,
        "vision": vision,
        "distance_sensor": distance_obs,
    }
    return state, velocity

def _calculate_distance(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    # [4] إصلاح حساب المسافة ليأخذ محور الـ Z في الاعتبار (3D Distance)
    dz = p1.get("z", 0.0) - p2.get("z", 0.0)
    return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2 + dz ** 2)

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
