# from __future__ import annotations

# import math
# import logging
# from typing import Any, Dict, List, Optional

# def classify_detection(label: str) -> str:
#     """تصنيف الكائنات المكتشفة لتبسيط السرد النصي"""
#     label = label.lower()
#     if any(x in label for x in ["person", "pedestrian", "people"]):
#         return "Human"
#     elif any(x in label for x in ["car", "bus", "truck", "van", "vehicle"]):
#         return "Heavy Vehicle"
#     elif any(x in label for x in ["bicycle", "motorcycle", "motor"]):
#         return "Light Vehicle"
#     elif "building" in label:
#         return "Structure"
#     return "Obstacle"

# class FlightNarrator:
#     """
#     Real-time flight narrator that converts raw drone state + RL action
#     into short, plain-English explanations of every navigation decision.

#     Mirrors the PathPlanner's obstacle thresholds (class-specific) so the
#     explanation accurately reflects *why* the planner chose avoidance vs. cruise.
#     """

#     # Must match PathPlanner.CLASS_THRESHOLDS exactly
#     CLASS_THRESHOLDS: Dict[str, float] = {
#         "person":    100.0,
#         "vehicle":   130.0,
#         "structure": 160.0,
#     }
#     DEFAULT_THRESHOLD = 160.0

#     # ── cardinal labels for human-friendly bearing ──────────────────────
#     _CARDINALS = [
#         "north", "north-northeast", "northeast", "east-northeast",
#         "east", "east-southeast", "southeast", "south-southeast",
#         "south", "south-southwest", "southwest", "west-southwest",
#         "west", "west-northwest", "northwest", "north-northwest",
#     ]

#     def __init__(self, logger: logging.Logger):
#         self.logger = logger

#     # ====================================================================
#     # Public API  –  call once per control-loop iteration
#     # ====================================================================
#     def narrate(
#         self,
#         action: str,
#         pose: Dict[str, float],
#         velocity: Dict[str, float],
#         target: Dict[str, float],
#         detections: List[Dict[str, Any]],
#         step: Optional[int] = None,
#     ) -> tuple[str, str, str]:
#         """
#         Returns (action_text, reason_text, status_text) and logs them.
#         """
#         dist_to_target = self._distance_3d(pose, target)
#         bearing = self._bearing_label(pose, target)
#         speed = math.sqrt(
#             velocity.get("x", 0) ** 2
#             + velocity.get("y", 0) ** 2
#             + velocity.get("z", 0) ** 2
#         )

#         obstacle_ahead, obstacle_class, obstacle_label = self._is_obstacle_ahead(detections)
#         closest = self._closest_detection(detections)

#         # 1. Action text
#         action_text = self._describe_action(action, obstacle_ahead)

#         # 2. Reason text
#         reason_text = self._describe_reason(
#             action, obstacle_ahead, obstacle_class, obstacle_label,
#             closest, dist_to_target, bearing, speed,
#         )

#         # 3. Status text
#         status_text = self._describe_status(detections, dist_to_target, bearing, pose)

#         step_prefix = f"[Step {step}] " if step is not None else ""
#         self.logger.info(
#             "\n%s🗣️ Action: %s\nReason: %s\nStatus: %s",
#             step_prefix, action_text, reason_text, status_text,
#         )

#         return action_text, reason_text, status_text

#     # ====================================================================
#     # Internal helpers
#     # ====================================================================

#         # ── action line ─────────────────────────────────────────────────────
#     def _describe_action(self, action, obstacle_ahead: bool = False) -> str:
#             """ترجمة أوامر السرعة المستمرة لنص مقروء"""
#             if isinstance(action, list) and len(action) == 3:
#                 vx, vy, vz = action
#                 return f"Adjusting velocity [vx: {vx:+.2f}, vy: {vy:+.2f}, vz: {vz:+.2f}]"
#             return f"Executing {action}"

#     # ── reason line ─────────────────────────────────────────────────────
#     def _describe_reason(
#         self,
#         action: str,
#         obstacle_ahead: bool,
#         obstacle_class: str,
#         obstacle_label: str,
#         closest: Optional[Dict[str, Any]],
#         dist_to_target: float,
#         bearing: str,
#         speed: float,
#     ) -> str:
#         if obstacle_ahead and closest:
#             conf = closest.get("confidence", 0) * 100
#             bbox = closest.get("bbox", [0, 0, 0, 0])
#             width = bbox[2] - bbox[0]
#             risk_label = {
#                 "person":    "HIGH RISK",
#                 "vehicle":   "MEDIUM RISK",
#                 "structure": "LOW RISK",
#             }
#             return (
#                 f'"{obstacle_label}" ({risk_label.get(obstacle_class, "UNKNOWN")}) '
#                 f"detected ahead with {conf:.0f}% confidence "
#                 f"(bbox width {width:.0f}px). "
#                 f"PathPlanner triggered {obstacle_class}-specific avoidance."
#             )

#         if action == "forward":
#             return (
#                 f"No obstacles in the forward field of view. "
#                 f"RL agent selected 'forward' — clear path toward target "
#                 f"{dist_to_target:.1f}m to the {bearing}."
#             )
#         if action == "turn_left":
#             return (
#                 f"RL agent chose to rotate left to align heading "
#                 f"with target {dist_to_target:.1f}m to the {bearing}."
#             )
#         if action == "turn_right":
#             return (
#                 f"RL agent chose to rotate right to align heading "
#                 f"with target {dist_to_target:.1f}m to the {bearing}."
#             )
#         if action in ("up", "down"):
#             verb = "ascend" if action == "up" else "descend"
#             return (
#                 f"RL agent chose to {verb} for better altitude positioning. "
#                 f"Target is {dist_to_target:.1f}m to the {bearing}."
#             )
#         # hover
#         return (
#             f"RL agent chose to hold position. "
#             f"Target is {dist_to_target:.1f}m to the {bearing}."
#         )

#     # ── status line ─────────────────────────────────────────────────────
#     def _describe_status(
#         self,
#         detections: List[Dict[str, Any]],
#         dist_to_target: float,
#         bearing: str,
#         pose: Dict[str, float],
#     ) -> str:
#         n = len(detections)
#         obs_part = (
#             "Path clear"
#             if n == 0
#             else f"{n} obstacle{'s' if n != 1 else ''} in view"
#         )
#         alt = abs(pose.get("z", 0))
#         return (
#             f"{obs_part}, target is {dist_to_target:.1f}m ahead "
#             f"to the {bearing}, altitude {alt:.1f}m."
#         )

#     # ── obstacle detection — class-specific thresholds ──────────────────
#     def _is_obstacle_ahead(
#         self, detections: List[Dict[str, Any]]
#     ) -> tuple[bool, str, str]:
#         """
#         Returns (is_blocked, risk_category, label).
#         Uses the same per-class thresholds as PathPlanner so the narration
#         matches the actual avoidance decision.
#         """
#         for det in detections:
#             bbox = det.get("bbox")
#             if not isinstance(bbox, list) or len(bbox) < 3:
#                 continue
#             try:
#                 x1, _, x2, *_ = bbox
#                 width    = max(float(x2) - float(x1), 0.0)
#                 label    = det.get("label", "unknown")
#                 category = classify_detection(label)
#                 threshold = self.CLASS_THRESHOLDS.get(category, self.DEFAULT_THRESHOLD)
#                 if width > threshold:
#                     return True, category, label
#             except (ValueError, TypeError):
#                 continue
#         return False, "", ""

#     def _closest_detection(
#         self, detections: List[Dict[str, Any]]
#     ) -> Optional[Dict[str, Any]]:
#         if not detections:
#             return None
#         return max(detections, key=lambda d: d.get("confidence", 0))

#     # ── geometry ────────────────────────────────────────────────────────
#     @staticmethod
#     def _distance_3d(a: Dict[str, float], b: Dict[str, float]) -> float:
#         return math.sqrt(
#             (a.get("x", 0) - b.get("x", 0)) ** 2
#             + (a.get("y", 0) - b.get("y", 0)) ** 2
#             + (a.get("z", 0) - b.get("z", 0)) ** 2
#         )

#     @classmethod
#     def _bearing_label(cls, frm: Dict[str, float], to: Dict[str, float]) -> str:
#         dx = to.get("x", 0) - frm.get("x", 0)
#         dy = to.get("y", 0) - frm.get("y", 0)
#         angle = math.degrees(math.atan2(dy, dx))
#         compass = (90 - angle) % 360
#         idx = int((compass + 11.25) / 22.5) % 16
#         return cls._CARDINALS[idx]

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional

def classify_detection(label: str) -> str:
    """تصنيف الكائنات المكتشفة لتبسيط السرد النصي"""
    label = label.lower()
    if any(x in label for x in ["person", "pedestrian", "people"]):
        return "Human"
    elif any(x in label for x in ["car", "bus", "truck", "van", "vehicle"]):
        return "Heavy Vehicle"
    elif any(x in label for x in ["bicycle", "motorcycle", "motor"]):
        return "Light Vehicle"
    elif "building" in label:
        return "Structure"
    return "Obstacle"


class FlightNarrator:
    """
    Real-time flight narrator that converts raw drone state + Continuous PPO action
    into short, plain-English explanations of every navigation decision.
    """

    # تم توحيد المفاتيح لتتطابق تماماً مع مخرجات classify_detection
    CLASS_THRESHOLDS: Dict[str, float] = {
        "Human":         100.0,
        "Light Vehicle": 120.0,
        "Heavy Vehicle": 130.0,
        "Structure":     160.0,
        "Obstacle":      150.0,
    }
    DEFAULT_THRESHOLD = 160.0

    # ── cardinal labels for human-friendly bearing ──────────────────────
    _CARDINALS = [
        "north", "north-northeast", "northeast", "east-northeast",
        "east", "east-southeast", "southeast", "south-southeast",
        "south", "south-southwest", "southwest", "west-southwest",
        "west", "west-northwest", "northwest", "north-northwest",
    ]

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    # ====================================================================
    # Public API
    # ====================================================================
    def narrate(
        self,
        action: List[float],
        pose: Dict[str, float],
        velocity: Dict[str, float],
        target: Dict[str, float],
        detections: List[Dict[str, Any]],
        step: Optional[int] = None,
    ) -> tuple[str, str, str]:
        """
        Returns (action_text, reason_text, status_text) and logs them.
        """
        dist_to_target = self._distance_3d(pose, target)
        bearing = self._bearing_label(pose, target)

        obstacle_ahead, obstacle_class, obstacle_label = self._is_obstacle_ahead(detections)
        closest = self._closest_detection(detections)

        # 1. Action text (translates the velocity vector)
        action_text = self._describe_action(action)

        # 2. Reason text (interprets WHY the vector was chosen)
        reason_text = self._describe_reason(
            action, obstacle_ahead, obstacle_class, obstacle_label,
            closest, dist_to_target, bearing
        )

        # 3. Status text
        status_text = self._describe_status(detections, dist_to_target, bearing, pose)

        step_prefix = f"[Step {step}] " if step is not None else ""
        self.logger.info(
            "\n%s🗣️ Action: %s\nReason: %s\nStatus: %s",
            step_prefix, action_text, reason_text, status_text,
        )

        return action_text, reason_text, status_text

    # ====================================================================
    # Internal helpers
    # ====================================================================

    def _describe_action(self, action: List[float]) -> str:
        """ترجمة أوامر السرعة المستمرة لنص مقروء"""
        if isinstance(action, list) and len(action) == 3:
            vx, vy, vz = action
            return f"Velocity command [vx: {vx:+.2f}, vy: {vy:+.2f}, vz: {vz:+.2f}] m/s"
        return f"Executing {action}"

    def _describe_reason(
        self,
        action: List[float],
        obstacle_ahead: bool,
        obstacle_class: str,
        obstacle_label: str,
        closest: Optional[Dict[str, Any]],
        dist_to_target: float,
        bearing: str,
    ) -> str:
        """تحليل متجه السرعة PPO واستنتاج نية الدرون"""
        vx, vy, vz = action if (isinstance(action, list) and len(action) == 3) else (0.0, 0.0, 0.0)
        
        # استنتاج نوع الحركة من الأرقام
        intent = "adjusting trajectory"
        if vz > 0.5:
            intent = "ascending"
        elif vz < -0.5:
            intent = "descending"
        elif abs(vx) > 0.5 or abs(vy) > 0.5:
            intent = "cruising"
        else:
            intent = "hovering / stabilizing"

        # ربط الحركة بالعوائق
        if obstacle_ahead and closest:
            conf = closest.get("confidence", 0) * 100
            bbox = closest.get("bbox", [0, 0, 0, 0])
            width = max(bbox[2] - bbox[0], 0)
            
            return (
                f'"{obstacle_label}" ({obstacle_class}) detected ahead '
                f'({conf:.0f}% conf, width: {width:.0f}px). '
                f'PPO agent is {intent} to avoid collision while maintaining target lock.'
            )

        # ربط الحركة بالهدف إذا كان المسار خالي
        return (
            f"Path is clear. PPO agent is {intent} towards the target "
            f"({dist_to_target:.1f}m to the {bearing})."
        )

    def _describe_status(
        self,
        detections: List[Dict[str, Any]],
        dist_to_target: float,
        bearing: str,
        pose: Dict[str, float],
    ) -> str:
        n = len(detections)
        obs_part = "Path clear" if n == 0 else f"{n} object{'s' if n != 1 else ''} tracked"
        alt = pose.get("z", 0)
        return (
            f"{obs_part} | Target: {dist_to_target:.1f}m {bearing} | Altitude: {alt:.1f}m"
        )

    def _is_obstacle_ahead(
        self, detections: List[Dict[str, Any]]
    ) -> tuple[bool, str, str]:
        for det in detections:
            bbox = det.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 3:
                continue
            try:
                x1, _, x2, *_ = bbox
                width    = max(float(x2) - float(x1), 0.0)
                label    = det.get("label", "unknown")
                category = classify_detection(label)
                threshold = self.CLASS_THRESHOLDS.get(category, self.DEFAULT_THRESHOLD)
                
                if width > threshold:
                    return True, category, label
            except (ValueError, TypeError):
                continue
        return False, "", ""

    def _closest_detection(
        self, detections: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not detections:
            return None
        return max(detections, key=lambda d: d.get("confidence", 0))

    @staticmethod
    def _distance_3d(a: Dict[str, float], b: Dict[str, float]) -> float:
        return math.sqrt(
            (a.get("x", 0) - b.get("x", 0)) ** 2
            + (a.get("y", 0) - b.get("y", 0)) ** 2
            + (a.get("z", 0) - b.get("z", 0)) ** 2
        )

    @classmethod
    def _bearing_label(cls, frm: Dict[str, float], to: Dict[str, float]) -> str:
        dx = to.get("x", 0) - frm.get("x", 0)
        dy = to.get("y", 0) - frm.get("y", 0)
        angle = math.degrees(math.atan2(dy, dx))
        compass = (90 - angle) % 360
        idx = int((compass + 11.25) / 22.5) % 16
        return cls._CARDINALS[idx]