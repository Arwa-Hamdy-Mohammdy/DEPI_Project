from __future__ import annotations
import logging
from typing import Any, Tuple, Optional, Dict
from enum import Enum


class FlightMode(Enum):
    NORMAL = "normal"
    SPRINT = "sprint"


from intelligence.rl_agent import classify_detection


class PathPlanner:
    """
    Converts perception + RL decisions into physical velocity/yaw commands.

    Navigation model (Yaw-based):
      - turn_left / turn_right  →  يلف الدرون في المكان (yaw rotation)
      - forward                 →  يمشي في اتجاه الـ yaw الحالي
      - up / down / hover       →  كما كانوا

    Object-Aware: thresholds و avoidance speeds مختلفة حسب نوع العائق.
    """

    # Per-class config: (bbox_width_threshold, avoidance_speed)
    CLASS_THRESHOLDS: Dict[str, Tuple[float, float]] = {
        "person":    (10.0, 1.5),
        "vehicle":   (12.0, 2.0),
        "structure": (8.0, 2.5),
    }
    DEFAULT_THRESHOLD = (8.0, 2.0)

    # يلف كام درجة في كل turn action
    YAW_STEP_DEGREES: float = 20.0
    # مدة الـ rotation بالثواني
    YAW_DURATION: float = 1.0

    def __init__(
        self,
        logger: logging.Logger,
        mode: FlightMode = FlightMode.NORMAL,
        default_duration: float = 2.0,
        obstacle_width_threshold: float = 160.0,
        speed_normal: float = 8.0,
        speed_sprint: float = 15.0,
        speed_avoidance: float = 2.0,
        speed_sprint_avoidance: float = 6.0,
        speed_vertical: float = 2.0,
    ):
        self.logger = logger
        self.mode = mode
        self.default_duration = default_duration
        self.obstacle_width_threshold = obstacle_width_threshold
        self.speed_normal = speed_normal
        self.speed_sprint = speed_sprint
        self.speed_avoidance = speed_avoidance
        self.speed_sprint_avoidance = speed_sprint_avoidance
        self.speed_vertical = speed_vertical

    def plan_next_move(
        self,
        detections: list[dict[str, Any]],
        rl_action: str,
        current_yaw: float = 0.0,
    ) -> dict:
        """
        Returns either:
          - {"type": "move",   "vx": …, "vy": …, "vz": …, "duration": …}
          - {"type": "rotate", "degrees": …, "duration": …}

        train_rl.py يتحقق من "type" ويستدعي controller.move أو controller.rotate_yaw.
        """
        obstacle_ahead, obstacle_class, obstacle_label = self._is_obstacle_ahead(detections)

        if self.mode == FlightMode.SPRINT:
            return self._plan_sprint(
                obstacle_ahead, obstacle_class, obstacle_label, rl_action, current_yaw
            )
        return self._plan_normal(
            obstacle_ahead, obstacle_class, obstacle_label, rl_action, current_yaw
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Normal mode
    # ─────────────────────────────────────────────────────────────────────────
    def _plan_normal(
        self,
        obstacle_ahead: bool,
        obstacle_class: str,
        obstacle_label: str,
        rl_action: str,
        current_yaw: float,
    ) -> dict:
        import math

        if obstacle_ahead:
            self.logger.info(
                "⚠️ %s detected ahead (%s risk). Path blocked!",
                obstacle_label, obstacle_class.upper()
            )
            # Safety Layer: Stop + Rotate
            if rl_action == "forward":
                self.logger.warning("Safety Override: Blocking 'forward'. Forcing 'turn_right'.")
                return self._build_rotate(+self.YAW_STEP_DEGREES)
            # If action is turn_left, turn_right, or hover, allow it to execute normally below

        # ── لا عائق (أو أكشن آمن): نفذ الـ action اللي اختاره الـ RL ──
        if rl_action == "forward":
            # يمشي في اتجاه الـ yaw الحالي (world-frame velocity)
            vx, vy = self._yaw_to_velocity(current_yaw, self.speed_normal)
            return self._build_move(vx=vx, vy=vy, vz=0.0)

        if rl_action == "turn_left":
            return self._build_rotate(-self.YAW_STEP_DEGREES)

        if rl_action == "turn_right":
            return self._build_rotate(+self.YAW_STEP_DEGREES)

        if rl_action == "up":
            return self._build_move(vx=0.0, vy=0.0, vz=-self.speed_vertical)

        if rl_action == "down":
            return self._build_move(vx=0.0, vy=0.0, vz=+self.speed_vertical)

        if rl_action == "hover":
            return self._build_move(vx=0.0, vy=0.0, vz=0.0)

        self.logger.debug("Unrecognized RL action '%s'. Hovering.", rl_action)
        return self._build_move(vx=0.0, vy=0.0, vz=0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Sprint mode
    # ─────────────────────────────────────────────────────────────────────────
    def _plan_sprint(
        self,
        obstacle_ahead: bool,
        obstacle_class: str,
        obstacle_label: str,
        rl_action: str,
        current_yaw: float,
    ) -> dict:
        if obstacle_ahead:
            _, avoidance_speed = self.CLASS_THRESHOLDS.get(obstacle_class, self.DEFAULT_THRESHOLD)
            self.logger.warning(
                "⚠️ %s detected in SPRINT mode (%s risk)! Avoidance at %.1f m/s.",
                obstacle_label, obstacle_class.upper(), avoidance_speed,
            )
            if rl_action == "up":
                return self._build_move(vx=0.0, vy=0.0, vz=-avoidance_speed)
            if rl_action == "turn_left":
                return self._build_rotate(-self.YAW_STEP_DEGREES)
            return self._build_rotate(+self.YAW_STEP_DEGREES)

        # مفيش عائق في sprint → فل سبيد في اتجاه الـ yaw
        self.logger.debug("Path clear. Sprinting forward.")
        vx, vy = self._yaw_to_velocity(current_yaw, self.speed_sprint)
        return self._build_move(vx=vx, vy=vy, vz=0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Obstacle detection
    # ─────────────────────────────────────────────────────────────────────────
    def _is_obstacle_ahead(
        self, detections: list[dict[str, Any]]
    ) -> Tuple[bool, str, str]:
        if not detections:
            return False, "", ""

        for obj in detections:
            bbox = obj.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 3:
                self.logger.warning("Malformed bounding box data: %s", bbox)
                continue
            try:
                x1, _, x2, *_ = bbox
                width    = max(float(x2) - float(x1), 0.0)
                label    = obj.get("label", "unknown")
                category = classify_detection(label)
                threshold, _ = self.CLASS_THRESHOLDS.get(category, self.DEFAULT_THRESHOLD)

                if width > threshold:
                    return True, category, label
            except (ValueError, TypeError) as e:
                self.logger.error("Failed to parse bbox %s: %s", bbox, e)
                continue

        return False, "", ""

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _yaw_to_velocity(yaw_deg: float, speed: float) -> Tuple[float, float]:
        """
        Converts a yaw angle (degrees) and speed into (vx, vy) in world frame.
        AirSim convention: yaw=0 → North (+X), yaw=90 → East (+Y).
        """
        import math
        rad = math.radians(yaw_deg)
        vx = speed * math.cos(rad)
        vy = speed * math.sin(rad)
        return vx, vy

    def _build_move(self, vx: float, vy: float, vz: float) -> dict:
        return {
            "type": "move",
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "duration": self.default_duration,
        }

    def _build_rotate(self, degrees: float) -> dict:
        return {
            "type":     "rotate",
            "degrees":  degrees,
            "duration": self.YAW_DURATION,
        }