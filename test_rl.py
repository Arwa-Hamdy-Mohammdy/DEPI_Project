import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from controllers.drone_controller import DroneController
from intelligence.rl_agent import RLAgent, RL_ACTION_TO_IDX
from intelligence.path_planner import PathPlanner, FlightMode
from services.vision_service import VisionService
from utils.logger import setup_logger

async def run_mission(target_object: Optional[str] = None):
    project_root = Path(__file__).resolve().parent
    logger = setup_logger(log_dir=project_root / "data" / "logs" / "inference")
    
    print("\n" + "="*50)
    print("🚁 DRONE MISSION CONTROL (INFERENCE MODE)")
    print("="*50)
    
    # Initialize Modules
    controller = DroneController(logger=logger)
    vision     = VisionService(project_root=project_root, logger=logger)
    planner    = PathPlanner(logger=logger, mode=FlightMode.NORMAL)
    agent      = RLAgent(
        logger=logger,
        model_path=project_root / "models" / "dqn_drone_model.pth",
        state_dim=18,
        action_dim=4,
    )
    agent.set_eval_mode()  # epsilon=0, no random exploration

    try:
        await controller.connect()
        if not controller.connected:
            print("❌ Error: Could not connect to AirSim.")
            return

        print(f"\n🚀 Mission Started: {f'Find {target_object.upper()}' if target_object else 'Explore Area'}")
        
        await controller.reset()
        await asyncio.sleep(1.0)
        await controller.takeoff()
        await controller.move(0, 0, -10.0, 2.0) # Ascend to safe height
        await asyncio.sleep(1.0)

        pose = await controller.get_pose()
        done = False
        step = 0

        while not done and step < 1000:
            # 1. Perception
            frames = await vision.get_frames()
            detections = await vision.get_fused_detections(frames["rgb"], frames["seg"])
            
            # Log detections as images (Drone Awareness visualization)
            if detections:
                await vision.save_detected_frame(frames["rgb"], detections)
            
            pose = await controller.get_pose()
            vel = await controller.get_velocity()
            yaw = await controller.get_yaw()
            
            # 2. Build State dict (passed directly — agent encodes internally)
            state = {
                "pose": pose, "target": None, "velocity": vel,
                "yaw": yaw, "detections": detections, "step": step,
                "search_target": target_object
            }
            
            # 3. Select Best Action (eval mode — no exploration)
            action_str = await agent.select_action(state)
            
            # 4. Plan & Visualize
            cmd = planner.plan_next_move(detections=detections, rl_action=action_str, current_yaw=yaw)
            decision_label = cmd.get("label", "NAVIGATING...")
            
            display_text = decision_label
            if target_object:
                # Check if target is actually found in current detections
                found = any(target_object.lower() in d["label"].lower() for d in detections)
                status = "FOUND! 🎯" if found else "SEARCHING..."
                display_text = f"OBJECT: {target_object.upper()} | {status} | {decision_label}"

            await controller.draw_decision(display_text, pose)
            
            # 5. Execute
            if cmd["type"] == "move":
                await controller.move(cmd["vx"], cmd["vy"], cmd["vz"], cmd["duration"])
            elif cmd["type"] == "rotate":
                await controller.rotate_yaw(cmd["degrees"], cmd["duration"])
            
            collision = await controller.has_collided()
            if collision:
                print("💥 Collision detected! Mission Aborted.")
                done = True
            
            # Check if target reached (visually large enough)
            if target_object:
                for d in detections:
                    if target_object.lower() in d["label"].lower():
                        bbox = d.get("bbox", [0,0,0,0])
                        size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if size > 60000:
                            print(f"🎯 Target '{target_object}' Reached!")
                            await controller.draw_decision(f"MISSION COMPLETE: {target_object.upper()} REACHED!", pose)
                            await asyncio.sleep(2.0)
                            done = True

            step += 1
            await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        print("\nMission cancelled.")
    except Exception as e:
        print(f"Error during mission: {e}")
    finally:
        await controller.disconnect()
        vision.shutdown()
        print("🏁 System Shutdown.")

if __name__ == "__main__":
    import sys
    print("\n[MISSION SELECTION]")
    print("1. General Exploration & Avoidance")
    print("2. Search & Find Specific Object")
    
    choice = input("\nEnter choice (1/2): ").strip()
    
    target = None
    if choice == '2':
        target = input("Enter object name (e.g., car, person, tree): ").strip().lower()
    
    try:
        asyncio.run(run_mission(target_object=target))
    except KeyboardInterrupt:
        pass
