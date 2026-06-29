from __future__ import annotations

import asyncio
import math
import time
import logging
import multiprocessing as mp
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List


from controllers.drone_controller import DroneController
from intelligence.path_planner import PathPlanner
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

    vision = VisionService(project_root=project_root, logger=logger)
    controller = DroneController(logger=logger)
    planner = PathPlanner(logger=logger)
    narrator = FlightNarrator(logger=logger)
    
    # [1] إصلاح الذاكرة: تعريف واحد فقط للـ Agent
    agent = RLAgent(
        logger=logger,
        model_path=project_root / "models" / "dqn_drone_model.pth",
    )
    agent.set_eval_mode()  # Disables random exploration for flight

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
        frame, current_pose, current_yaw = await asyncio.gather(
            vision.get_frame(),
            controller.get_pose(),
            controller.get_yaw(),
        )

        detections = await vision.detect_objects(frame)
        current_time = time.time()
        
        # [3] توحيد بناء مساحة الحالة (State) وحساب السرعة
        state, velocity = _build_consistent_state(current_pose, previous_pose, dummy_target, detections, current_time, last_time, current_yaw)

        action = await agent.select_action(state)
        movement_cmd = planner.plan_next_move(detections=detections, rl_action=action, current_yaw=current_yaw)

        await _execute_movement(controller, movement_cmd)
        logger.info("Step %s completed. action=%s", step, action)

        narrator.narrate(
            action=action, pose=current_pose, velocity=velocity,
            target=dummy_target, detections=detections, step=step,
        )

        asyncio.create_task(vision.save_detected_frame(frame, detections))
        
        previous_pose = current_pose
        last_time = current_time
        await asyncio.sleep(delay)

async def _run_goal_mission(
    controller: DroneController, vision: VisionService, planner: PathPlanner,
    agent: RLAgent, narrator: FlightNarrator, logger: logging.Logger, target_distance: float, delay: float,
) -> None:
    
    start_pose = await controller.get_pose()
    # نحسب هدفاً تقريبياً بناءً على المسافة للحفاظ على دقة الـ State
    estimated_target = {"x": start_pose["x"] + target_distance, "y": start_pose["y"], "z": start_pose.get("z", -10.0)}
    
    previous_pose = start_pose
    last_time = time.time()
    reached_target = False

    while not reached_target:
        current_pose, frame, current_yaw = await asyncio.gather(
            controller.get_pose(),
            vision.get_frame(),
            controller.get_yaw(),
        )
        
        distance_covered = _calculate_distance(start_pose, current_pose)
        detections = await vision.detect_objects(frame)
        current_time = time.time()

        state, velocity = _build_consistent_state(current_pose, previous_pose, estimated_target, detections, current_time, last_time, current_yaw)
        action = await agent.select_action(state)

        if distance_covered < target_distance:
            movement_cmd = planner.plan_next_move(detections=detections, rl_action=action, current_yaw=current_yaw)
            await _execute_movement(controller, movement_cmd)
            narrator.narrate(
                action=action, pose=current_pose, velocity=velocity,
                target=estimated_target, detections=detections,
            )
            print(f"Distance covered: {distance_covered:.2f} / {target_distance}m", end="\r")
        else:
            logger.info("\n✅ Goal Reached! %s meters covered.", target_distance)
            reached_target = True

        asyncio.create_task(vision.save_detected_frame(frame, detections))
        
        previous_pose = current_pose
        last_time = current_time
        await asyncio.sleep(delay)

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
        current_pose, frame, current_yaw = await asyncio.gather(
            controller.get_pose(),
            vision.get_frame(),
            controller.get_yaw(),
        )
        
        distance_to_target = _calculate_distance(current_pose, target_pose)

        if distance_to_target <= acceptance_radius:
            reached_target = True
            break

        detections = await vision.detect_objects(frame)
        current_time = time.time()
        
        # 2. previous_pose is now guaranteed to exist on the first pass
        state, velocity = _build_consistent_state(
            current_pose, previous_pose, target_pose, detections, current_time, last_time, current_yaw
        )
        
        action = await agent.select_action(state)
        movement_cmd = planner.plan_next_move(detections=detections, rl_action=action, current_yaw=current_yaw)
        
        await _execute_movement(controller, movement_cmd)

        narrator.narrate(
            action=action, pose=current_pose, velocity=velocity,
            target=target_pose, detections=detections,
        )

        asyncio.create_task(vision.save_detected_frame(frame, detections))
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
async def _execute_movement(controller: DroneController, movement_cmd: Dict[str, Any]) -> None:
    """ينفذ أمر الحركة الصحيح — rotate_yaw أو move — بناءً على نوع الأمر."""
    if movement_cmd.get("type") == "rotate":
        await controller.rotate_yaw(movement_cmd["degrees"], movement_cmd.get("duration", 1.0))
    else:
        await controller.move(
            movement_cmd["vx"], movement_cmd["vy"],
            movement_cmd["vz"], movement_cmd["duration"]
        )

def _build_consistent_state(
    current_pose: Dict[str, float], 
    previous_pose: Dict[str, float], 
    target_pose: Dict[str, float], 
    detections: List[Any], 
    current_time: float, 
    last_time: float,
    yaw: float = 0.0,
) -> tuple[Dict[str, Any], Dict[str, float]]:
    """يضمن إرسال مساحة الحالة (State Space) بنفس الشكل الذي تدرب عليه النموذج دائماً"""
    dt = current_time - last_time
    if dt <= 0:
        dt = 0.001
        
    velocity = {
        "x": (current_pose["x"] - previous_pose["x"]) / dt,
        "y": (current_pose["y"] - previous_pose["y"]) / dt,
        "z": (current_pose.get("z", 0) - previous_pose.get("z", 0)) / dt,
    }
    
    state = {
        "pose": current_pose,
        "target": target_pose,
        "detections": detections,
        "velocity": velocity,
        "yaw": yaw,
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