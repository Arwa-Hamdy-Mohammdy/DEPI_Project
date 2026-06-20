"""
train_rl.py — نظام التدريب الرئيسي للدرون الذكي
=================================================
السيناريو الكامل:
  1. اختيار المهمة (Exploration أو Search & Find)
  2. الاتصال بـ AirSim والإقلاع الآمن لـ 10m
  3. حلقة القرار: Perception → Encoding → Action → Safety Layer → Reward
  4. نظام المكافأة مع عقاب الاصطدام وتفادي العوائق
  5. حفظ الموديل والصور تلقائياً
"""

import asyncio
import random
import math
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# ─── Imports من ملفات المشروع ───────────────────────────────────────────────
from controllers.drone_controller import DroneController
from intelligence.rl_agent import RLAgent, RL_ACTION_TO_IDX
from intelligence.path_planner import PathPlanner, FlightMode
from services.vision_service import VisionService
from utils.logger import setup_logger

# ═══════════════════════════════════════════════════════════════════════════════
# دالة المكافأة
# ═══════════════════════════════════════════════════════════════════════════════
def compute_reward(
    current_pose:       Dict[str, float],
    previous_pose:      Dict[str, float],
    target_pose:        Optional[Dict[str, float]],
    collision:          bool,
    step:               int,
    max_steps:          int = 500,
    current_yaw:        float = 0.0,
    action_idx:         int = 3,
    last_action_idx:    Optional[int] = None,
    safety_override:    bool = False,
    detections:         List[Dict[str, Any]] = [],
    search_target_label: Optional[str] = None
) -> Tuple[float, bool]:
    """
    حساب المكافأة لكل step.
    يدعم:
      - Coordinate Navigation: يكافئ الاقتراب من إحداثيات هدف.
      - Search & Find:         يكافئ رؤية الهدف المرئي والاقتراب منه.
    """
    # ── اصطدام ──────────────────────────────────────────────────────────────
    if collision:
        return -30.0, True   # عقاب كبير + إنهاء الـ episode

    reward = 0.0

    # ── Search & Find ────────────────────────────────────────────────────────
    if search_target_label:
        target_size = 0.0
        found_target = False
        for det in detections:
            if search_target_label.lower() in det.get("label", "").lower():
                found_target = True
                bbox = det.get("bbox", [0, 0, 0, 0])
                size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                target_size = max(target_size, size)

        if found_target:
            reward += 10.0
            reward += (target_size / 50000.0) * 5.0
            if target_size > 60000:
                return 100.0, True  # وصل للهدف المرئي!
        else:
            reward -= 0.1  # عقاب خفيف لكل خطوة بدون رؤية الهدف

    # ── Coordinate Navigation ────────────────────────────────────────────────
    if target_pose:
        prev_dist = math.sqrt(
            (previous_pose["x"] - target_pose["x"]) ** 2 +
            (previous_pose["y"] - target_pose["y"]) ** 2
        )
        curr_dist = math.sqrt(
            (current_pose["x"] - target_pose["x"]) ** 2 +
            (current_pose["y"] - target_pose["y"]) ** 2
        )
        progress = prev_dist - curr_dist
        reward += (progress / 5.0) * 2.0

        # مكافأة الاتجاه نحو الهدف
        dx = target_pose["x"] - current_pose["x"]
        dy = target_pose["y"] - current_pose["y"]
        target_bearing = math.degrees(math.atan2(dy, dx))
        angle_error = (target_bearing - current_yaw + 180) % 360 - 180
        alignment = math.cos(math.radians(angle_error))
        reward += 0.5 * alignment

        if curr_dist < 5.0:
            return 50.0, True   # وصل للهدف!

    # ── عقوبات سلوكية ────────────────────────────────────────────────────────
    if safety_override:
        reward -= 0.5   # Safety override ممكن بس ليس مثالي

    # عقاب الاهتزاز يمين-يسار في نفس المكان
    if last_action_idx is not None:
        if (action_idx == 1 and last_action_idx == 2) or \
           (action_idx == 2 and last_action_idx == 1):
            reward -= 2.0

    if action_idx == 3:
        reward -= 0.5  # عقاب التحوم (hover penalty)
    reward -= 0.05     # عقاب خطوة (step penalty → يشجع على الوصول بسرعة)

    if step >= max_steps:
        return reward - 5.0, True   # انتهى الوقت

    return max(min(reward, 20.0), -20.0), False


# ═══════════════════════════════════════════════════════════════════════════════
# Mission Selection
# ═══════════════════════════════════════════════════════════════════════════════
def ask_mission() -> Optional[str]:
    """
    يسأل المستخدم في الـ Terminal عن نوع المهمة.
    بيرجع:
      - None         → Exploration (يكتشف بحرية)
      - str (label)  → Search & Find (يدور على object محدد)
    """
    print("\n" + "═" * 60)
    print("  🚁  DRONE AI — Mission Selection")
    print("═" * 60)
    print("  [1]  Exploration Mode")
    print("       Drone explores the environment and avoids obstacles automatically")
    print()
    print("  [2]  Search & Find Mode")
    print("       Drone searches for a specific target (e.g. car / person / tree)")
    print("═" * 60)

    while True:
        choice = input("  Choose (1 or 2): ").strip()
        if choice == "1":
            print("\n  ✅ Exploration Mode selected. Good luck!\n")
            return None
        elif choice == "2":
            target = input("  Enter the target name (e.g., car): ").strip().lower()
            if target:
                print(f"\n  ✅ Search & Find Mode: Hunting for '{target}'!\n")
                return target
            else:
                print("  ⚠️  You must enter a target name. Try again.")
        else:
            print("  ⚠️  Choose 1 or 2 only.")


# ═══════════════════════════════════════════════════════════════════════════════
# الحلقة الرئيسية للتدريب
# ═══════════════════════════════════════════════════════════════════════════════
async def train(
    num_episodes: int = 1000,
    target_object: Optional[str] = None,
    airsim_ip: str = "127.0.0.1",
    airsim_port: int = 41451,
):
    project_root = Path(__file__).resolve().parent
    logger = setup_logger(log_dir=project_root / "data" / "logs" / "training")
    logger.info(
        "🚀 Starting Autonomous Drone RL Training "
        f"(Mission: {'Search ' + target_object if target_object else 'Exploration'})"
    )

    # ── تجهيز المجلدات والموديل ──────────────────────────────────────────────
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "dqn_drone_model.pth"

    # ── تهيئة الـ modules ──────────────────────────────────────────────────
    controller = DroneController(logger=logger)
    vision     = VisionService(
        project_root=project_root,
        logger=logger,
        airsim_ip=airsim_ip,
        airsim_port=airsim_port,
    )
    planner    = PathPlanner(
        logger=logger,
        mode=FlightMode.NORMAL,
        speed_normal=5.0,
        speed_vertical=3.0,
    )
    agent = RLAgent(
        logger=logger,
        model_path=model_path,
        state_dim=18,
        action_dim=4,
        epsilon_decay=0.995,
    )

    # حفظ نقطة بداية
    agent.save(model_path)
    logger.info(f"Initial model checkpoint created at {model_path}")

    try:
        # ── الاتصال بـ AirSim ──────────────────────────────────────────────
        print("\nConnecting to AirSim...")
        await controller.connect(timeout=30, ip=airsim_ip, port=airsim_port)
        if not controller.connected:
            print("\n" + "!" * 60)
            print(f"  ❌ Failed to connect to AirSim at {airsim_ip}:{airsim_port}!")
            if controller.last_connect_error:
                print(f"  ▶ Details: {controller.last_connect_error}")
            print("  1. Make sure AirSim is running and the simulator window is open.")
            print("  2. Make sure you chose 'Multirotor' when prompted by AirSim.")
            print("  3. If AirSim is running on another machine, pass --airsim-ip <ip>.")
            print("  4. If AirSim uses a nonstandard RPC port, pass --airsim-port <port>.")
            print("  5. Verify Windows Firewall is not blocking AirSim RPC.")
            print("!" * 60)
            return
        print("✅ Connected to AirSim!")

        # ── حلقة الـ Episodes ────────────────────────────────────────────────
        for ep in range(num_episodes):
            try:
                print(f"\n{'═' * 60}")
                print(f"  EPISODE {ep + 1}/{num_episodes}  |  ε = {agent.epsilon:.4f}")
                if target_object:
                    print(f"  🎯 MISSION: Search & Find → '{target_object.upper()}'")
                else:
                    print(f"  🗺️  MISSION: Exploration")
                print("═" * 60)

                # ── Reset ──────────────────────────────────────────────────
                print("  Resetting environment...")
                await controller.reset()
                await asyncio.sleep(2.0)

                # ── Takeoff ────────────────────────────────────────────────
                print("  Taking off...")
                await controller.takeoff()
                await asyncio.sleep(2.0)

                # ── الصعود لارتفاع آمن (10m → Z = -10 في AirSim) ─────────
                print("  Climbing to safe altitude (10m)...")
                await controller.move(0.0, 0.0, -5.0, 3.0)
                await asyncio.sleep(2.0)

                # التحقق من الارتفاع
                pose = await controller.get_pose()
                if pose["z"] > -2.0:
                    print(f"  ⚠️ Altitude check failed (Z={pose['z']:.1f}m). Retrying...")
                    await controller.takeoff()
                    await asyncio.sleep(1.0)
                    await controller.move(0.0, 0.0, -5.0, 3.0)
                    await asyncio.sleep(2.0)
                    pose = await controller.get_pose()

                print(f"  ✅ Airborne at Z={pose['z']:.1f}m")

                # ── تحديد الهدف ────────────────────────────────────────────
                target_pose = None
                if not target_object:
                    target_pose = {
                        "x": random.uniform(-60, 60),
                        "y": random.uniform(-60, 60),
                        "z": -10.0
                    }
                    print(f"  🎯 Target: ({target_pose['x']:.1f}, {target_pose['y']:.1f})")

                # ── الـ State الأولي ────────────────────────────────────────
                pose        = await controller.get_pose()
                vel         = await controller.get_velocity()
                yaw         = await controller.get_yaw()
                frames      = await vision.get_frames()
                detections  = await vision.get_fused_detections(frames["rgb"], frames["seg"])

                state = {
                    "pose":          pose,
                    "target":        target_pose if target_pose else pose,
                    "velocity":      vel,
                    "yaw":           yaw,
                    "detections":    detections,
                    "step":          0,
                    "search_target": target_object,
                }

                total_reward  = 0.0
                done          = False
                step          = 0
                last_action_idx = None

                # ══════════════════════════════════════════════════════════
                # حلقة القرار الرئيسية
                # ══════════════════════════════════════════════════════════
                while not done and step < 500:
                    try:
                        # 1. اختيار الحركة (RL or ε-greedy)
                        action_str = await agent.select_action(state)
                        action_idx = RL_ACTION_TO_IDX.get(action_str, 3)  # for reward calc

                        # 2. Safety Layer + تحويل الـ action لأمر فيزيائي
                        cmd = planner.plan_next_move(detections=detections, rl_action=action_str, current_yaw=yaw)
                        safety_override  = cmd.get("safety_override", False)
                        decision_label   = cmd.get("label", "THINKING...")

                        # نص العرض على شاشة AirSim
                        if target_object:
                            display_text = f"SEARCH:{target_object.upper()} | {decision_label}"
                        else:
                            display_text = decision_label

                        # عرض القرار على HUD
                        try:
                            await controller.draw_decision(display_text, pose)
                        except Exception as e:
                            logger.debug(f"HUD draw failed: {e}")

                        # 3. تنفيذ الأمر
                        if cmd["type"] == "move":
                            await controller.move(
                                cmd["vx"], cmd["vy"], cmd["vz"], cmd["duration"]
                            )
                        elif cmd["type"] == "rotate":
                            await controller.rotate_yaw(
                                cmd["degrees"], cmd["duration"]
                            )

                        # 4. الرصد بعد الحركة
                        new_pose       = await controller.get_pose()
                        new_vel        = await controller.get_velocity()
                        new_yaw        = await controller.get_yaw()
                        collision      = await controller.has_collided()

                        new_frames     = await vision.get_frames()
                        new_detections = await vision.get_fused_detections(
                            new_frames["rgb"], new_frames["seg"]
                        )

                        # حفظ صور الرصد كل 5 خطوات
                        if new_detections and step % 5 == 0:
                            try:
                                await vision.save_detected_frame(new_frames["rgb"], new_detections)
                            except Exception as e:
                                logger.debug(f"Save frame failed: {e}")

                        # 5. حساب المكافأة
                        reward, done = compute_reward(
                            new_pose, pose, target_pose, collision, step,
                            current_yaw=new_yaw,
                            action_idx=action_idx,
                            last_action_idx=last_action_idx,
                            safety_override=safety_override,
                            detections=new_detections,
                            search_target_label=target_object,
                        )

                        # 6. الـ State التالي
                        next_state = {
                            "pose":          new_pose,
                            "target":        target_pose if target_pose else new_pose,
                            "velocity":      new_vel,
                            "yaw":           new_yaw,
                            "detections":    new_detections,
                            "step":          step + 1,
                            "search_target": target_object,
                        }

                        # 7. تخزين التجربة والتدريب
                        agent.remember(state, action_str, reward, next_state, done)
                        loss = agent.train_step()

                        # تحديث المتغيرات
                        state          = next_state
                        pose           = new_pose
                        yaw            = new_yaw
                        detections     = new_detections
                        last_action_idx = action_idx
                        total_reward   += reward
                        step           += 1

                        # طباعة لحظية في الـ Terminal
                        if step % 10 == 0:
                            loss_str = f"{loss:.4f}" if loss is not None else "N/A "
                            bar_len = 20
                            prog = int((step / 500) * bar_len)
                            bar = "█" * prog + "░" * (bar_len - prog)
                            print(
                                f"\r  [{bar}] "
                                f"Step:{step:3d} | "
                                f"R:{total_reward:7.2f} | "
                                f"Loss:{loss_str} | "
                                f"ε:{agent.epsilon:.3f} | "
                                f"{decision_label[:25]}",
                                end="",
                                flush=True
                            )

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"Error in step {step}: {e}", exc_info=True)
                        break

                # ── نهاية الـ Episode ──────────────────────────────────────
                agent.update_epsilon()
                print(f"\n  {'─' * 58}")
                print(f"  ✅ Ep {ep + 1} done | Steps:{step} | "
                      f"Total Reward:{total_reward:7.2f} | ε:{agent.epsilon:.4f}")
                logger.info(
                    f"Episode {ep + 1}/{num_episodes} | "
                    f"Steps: {step} | Reward: {total_reward:.2f} | "
                    f"Epsilon: {agent.epsilon:.4f}"
                )

                # حفظ الموديل كل 10 episodes
                if (ep + 1) % 10 == 0:
                    agent.save(model_path)
                    print(f"  💾 Model checkpoint saved (ep {ep + 1})")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Episode {ep + 1} failed: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    except asyncio.CancelledError:
        logger.info("Training cancelled by user.")
    except Exception as e:
        logger.exception(f"Fatal error in training loop: {e}")
    finally:
        logger.info("Shutting down...")
        try:
            agent.save(model_path)
            print("\n  💾 Final model checkpoint saved.")
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
        try:
            await controller.disconnect()
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
        try:
            vision.shutdown()
        except Exception as e:
            logger.error(f"Vision shutdown error: {e}")
        logger.info("🏁 Shutdown complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Drone RL Training. For autonomous test/demo use: python main.py"
    )
    parser.add_argument(
        "--target", type=str, default=None,
        metavar="OBJECT",
        help="Target object label for Search mode (e.g. car, person, tree). "
             "Omit for Exploration mode (default)."
    )
    parser.add_argument(
        "--episodes", type=int, default=1000,
        help="Number of training episodes (default: 1000)"
    )
    parser.add_argument(
        "--airsim-ip", type=str, default="127.0.0.1",
        help="AirSim RPC host/IP (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--airsim-port", type=int, default=41451,
        help="AirSim RPC port (default: 41451)"
    )
    args = parser.parse_args()

    mode_str = f"Search & Find [{args.target}]" if args.target else "Exploration"
    print(f"  Training Mode : {mode_str}")
    print(f"  Episodes      : {args.episodes}")
    print(f"  (For live demo with mission selection use: python main.py)")
    print()

    try:
        asyncio.run(train(
            num_episodes=args.episodes,
            target_object=args.target,
            airsim_ip=args.airsim_ip,
            airsim_port=args.airsim_port,
        ))
    except KeyboardInterrupt:
        print("\n  [Ctrl+C] Shutdown requested.")
