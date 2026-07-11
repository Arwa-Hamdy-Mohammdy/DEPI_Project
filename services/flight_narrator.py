from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional

OBJECT_RISK_CATEGORIES: Dict[str, str] = {
    "person": "person", "pedestrian": "person", "child": "person",
    "worker": "person", "cyclist": "person",
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
    "motorcycle": "vehicle", "bicycle": "vehicle", "van": "vehicle",
    "boat": "vehicle", "train": "vehicle",
    "building": "structure", "house": "structure", "wall": "structure",
    "fence": "structure", "bridge": "structure", "tower": "structure",
    "pole": "structure", "traffic light": "structure", "stop sign": "structure",
    "fire hydrant": "structure", "bench": "structure", "tree": "structure",
}
DEFAULT_RISK_CATEGORY = "structure"

def classify_detection(label: str) -> str:
    """Map a YOLO label to a risk category: 'person', 'vehicle', or 'structure'."""
    return OBJECT_RISK_CATEGORIES.get(label.lower().strip(), DEFAULT_RISK_CATEGORY)

class FlightNarrator:
    """
    Real-time flight narrator that converts raw drone state + RL action
    into short, plain-English explanations of every navigation decision.

    Mirrors the PathPlanner's obstacle thresholds (class-specific) so the
    explanation accurately reflects *why* the planner chose avoidance vs. cruise.
    """

    # Must match PathPlanner.CLASS_THRESHOLDS exactly
    CLASS_THRESHOLDS: Dict[str, float] = {
        "person":    100.0,
        "vehicle":   130.0,
        "structure": 160.0,
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
    # Public API  –  call once per control-loop iteration
    # ====================================================================
    def narrate(
        self,
        action: Any,
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
        speed = math.sqrt(
            velocity.get("x", 0) ** 2
            + velocity.get("y", 0) ** 2
            + velocity.get("z", 0) ** 2
        )

        obstacle_ahead, obstacle_class, obstacle_label = self._is_obstacle_ahead(detections)
        closest = self._closest_detection(detections)

        # 1. Action text
        action_text = self._describe_action(action, obstacle_ahead)

        # 2. Reason text
        reason_text = self._describe_reason(
            action, obstacle_ahead, obstacle_class, obstacle_label,
            closest, dist_to_target, bearing, speed,
        )

        # 3. Status text
        status_text = self._describe_status(detections, dist_to_target, bearing, pose)

        step_prefix = f"[Step {step}] " if step is not None else ""
        self.logger.info(
            "\n" + "-" * 60 + "\n%s🗣️ Action: %s\nReason: %s\nStatus: %s",
            step_prefix, action_text, reason_text, status_text,
        )

        return action_text, reason_text, status_text

    # ====================================================================
    # Internal helpers
    # ====================================================================

    # ── action line ─────────────────────────────────────────────────────
    def _describe_action(self, action: Any, obstacle_ahead: bool) -> str:
        if obstacle_ahead:
            return "Obstacle in view — applying continuous velocity adjustments"

        try:
            vx, vy, vz = float(action[0]), float(action[1]), float(action[2])
            return f"Executing continuous velocities: vx={vx:.1f}, vy={vy:.1f}, vz={vz:.1f}"
        except:
            return f"Executing maneuver {action}"

    # ── reason line ─────────────────────────────────────────────────────
    def _describe_reason(
        self,
        action: Any,
        obstacle_ahead: bool,
        obstacle_class: str,
        obstacle_label: str,
        closest: Optional[Dict[str, Any]],
        dist_to_target: float,
        bearing: str,
        speed: float,
    ) -> str:
        if obstacle_ahead and closest:
            conf = closest.get("confidence", 0) * 100
            bbox = closest.get("bbox", [0, 0, 0, 0])
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            # Assume standard AirSim resolution 640x480 (307,200 pixels)
            area_ratio = area / 307200.0
            
            if area_ratio >= 0.20:
                risk_level = "HIGH RISK"
            elif area_ratio >= 0.05:
                risk_level = "MEDIUM RISK"
            else:
                risk_level = "LOW RISK"
                
            return (
                f'"{obstacle_label}" ({risk_level}) '
                f"detected ahead with {conf:.0f}% confidence "
                f"(bbox area {area_ratio*100:.1f}%). "
                f"PPO agent adjusting velocities for avoidance."
            )

        return (
            f"No critical obstacles in the forward field of view. "
            f"PPO agent selected continuous velocity vector to progress toward target "
            f"{dist_to_target:.1f}m to the {bearing}."
        )

    # ── status line ─────────────────────────────────────────────────────
    def _describe_status(
        self,
        detections: List[Dict[str, Any]],
        dist_to_target: float,
        bearing: str,
        pose: Dict[str, float],
    ) -> str:
        n = len(detections)
        obs_part = (
            "Path clear"
            if n == 0
            else f"{n} obstacle{'s' if n != 1 else ''} in view"
        )
        alt = abs(pose.get("z", 0))
        return (
            f"{obs_part}, target is {dist_to_target:.1f}m ahead "
            f"to the {bearing}, altitude {alt:.1f}m."
        )

    # ── obstacle detection — class-specific thresholds ──────────────────
    def _is_obstacle_ahead(
        self, detections: List[Dict[str, Any]]
    ) -> tuple[bool, str, str]:
        """
        Returns (is_blocked, risk_category, label).
        Uses the same per-class thresholds as PathPlanner so the narration
        matches the actual avoidance decision.
        """
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

    # ── geometry ────────────────────────────────────────────────────────
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