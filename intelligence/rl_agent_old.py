from __future__ import annotations

import asyncio
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# Action Space  —  يلف ثم يمشي (Yaw-based navigation)
# ═══════════════════════════════════════════════════════════════════════════════
class DroneAction(str, Enum):
    FORWARD    = "forward"
    TURN_LEFT  = "turn_left"   # يلف يسار (CCW)
    TURN_RIGHT = "turn_right"  # يلف يمين (CW)
    UP         = "up"
    DOWN       = "down"
    HOVER      = "hover"


ACTION_LIST: List[str] = [a.value for a in DroneAction]
ACTION_TO_IDX: Dict[str, int] = {a.value: i for i, a in enumerate(DroneAction)}
IDX_TO_ACTION: Dict[int, str] = {i: a.value for i, a in enumerate(DroneAction)}

# RL-specific action space (4 actions — no altitude during navigation training)
RL_ACTIONS: List[str] = ["forward", "turn_left", "turn_right", "hover"]
RL_ACTION_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(RL_ACTIONS)}
RL_IDX_TO_ACTION: Dict[int, str] = {i: a for i, a in enumerate(RL_ACTIONS)}


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Network  —  Dueling DQN
# ═══════════════════════════════════════════════════════════════════════════════
class DQNNetwork(nn.Module):
    """Deep Q-Network for discrete drone navigation."""

    def __init__(self, state_dim: int, action_dim: int = 6):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# ═══════════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity: int = 100_000):
        self.memory: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.memory)


# ═══════════════════════════════════════════════════════════════════════════════
# Object Classification  (YOLO label → risk category)
# ═══════════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════════
# State Encoder  —  18-dim
# ═══════════════════════════════════════════════════════════════════════════════
class StateEncoder:
    """
    Converts raw drone observations into a normalized state vector.

    18-dim vector layout:
      [0-2]   dx, dy, dz              — relative target position (normalized)
      [3-5]   vx, vy, vz              — normalized velocity
      [6]     yaw_norm                — current yaw in [-1, 1]  (÷180°)
      [7]     angle_to_target_norm    — signed angle error drone→target ÷ 180°
                                        (0 = facing target, ±1 = facing away)
      [8-10]  proximity, center_x, size — closest obstacle
      [11-12] is_person,  person_proximity
      [13-14] is_vehicle, vehicle_proximity
      [15-16] is_structure, structure_proximity
      [17]    step_norm               — normalized mission time
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        image_width: float = 640.0,
        image_height: float = 480.0,
    ):
        self.max_distance = max_distance
        self.image_width = image_width
        self.image_height = image_height
        self.state_dim = 18  # ← زودنا 2 بدل 16

    def encode(self, state: Dict[str, Any]) -> np.ndarray:
        """Returns a (18,) numpy array."""
        pose   = state.get("pose")   or state.get("drone_pose") or {"x": 0.0, "y": 0.0, "z": 0.0}
        target = state.get("target") or pose

        dx = (target.get("x", 0.0) - pose.get("x", 0.0)) / self.max_distance
        dy = (target.get("y", 0.0) - pose.get("y", 0.0)) / self.max_distance
        dz = (target.get("z", 0.0) - pose.get("z", 0.0)) / self.max_distance

        vel = state.get("velocity") or {"x": 0.0, "y": 0.0, "z": 0.0}
        vx = vel.get("x", 0.0) / 10.0
        vy = vel.get("y", 0.0) / 10.0
        vz = vel.get("z", 0.0) / 10.0

        # ── Yaw features ──
        yaw_deg = state.get("yaw", 0.0)          # degrees, يجي من drone_controller
        yaw_norm = yaw_deg / 180.0               # [-1, 1]

        # الزاوية من الـ yaw الحالي لاتجاه التارجت
        # target_bearing: الزاوية اللازمة عشان تواجه التارجت (degrees)
        import math
        target_dx = target.get("x", 0.0) - pose.get("x", 0.0)
        target_dy = target.get("y", 0.0) - pose.get("y", 0.0)
        target_bearing = math.degrees(math.atan2(target_dy, target_dx))

        # angle_error: الفرق بين اتجاهك والتارجت، في نطاق [-180, 180]
        angle_error = target_bearing - yaw_deg
        angle_error = (angle_error + 180) % 360 - 180   # wrap
        angle_to_target_norm = angle_error / 180.0       # [-1, 1]

        # ── Obstacle features ──
        detections: List[Dict[str, Any]] = state.get("detections", [])
        if detections:
            best = max(detections, key=lambda d: d.get("confidence", 0.0))
            bbox = best.get("bbox", [0.0, 0.0, 0.0, 0.0])
            x1, y1, x2, y2 = [float(v) for v in bbox]
            center_x = ((x1 + x2) / 2.0) / self.image_width
            size = ((x2 - x1) * (y2 - y1)) / (self.image_width * self.image_height)
            obstacle_proximity = min(best.get("confidence", 0.0), 1.0)
        else:
            center_x = 0.5
            size = 0.0
            obstacle_proximity = 0.0

        person_prox = vehicle_prox = structure_prox = 0.0
        has_person = has_vehicle = has_structure = 0.0

        for det in detections:
            label    = det.get("label", "")
            conf     = min(det.get("confidence", 0.0), 1.0)
            category = classify_detection(label)

            if category == "person":
                has_person = 1.0
                person_prox = max(person_prox, conf)
            elif category == "vehicle":
                has_vehicle = 1.0
                vehicle_prox = max(vehicle_prox, conf)
            else:
                has_structure = 1.0
                structure_prox = max(structure_prox, conf)

        step_norm = min(state.get("step", 0), 1000) / 1000.0

        vector = np.array([
            dx, dy, dz,                          # [0-2]
            vx, vy, vz,                          # [3-5]
            yaw_norm,                            # [6]
            angle_to_target_norm,                # [7]  ← الجديد المهم
            obstacle_proximity, center_x, size,  # [8-10]
            has_person,  person_prox,            # [11-12]
            has_vehicle, vehicle_prox,           # [13-14]
            has_structure, structure_prox,       # [15-16]
            step_norm,                           # [17]
        ], dtype=np.float32)

        return vector


# ═══════════════════════════════════════════════════════════════════════════════
# RL Agent
# ═══════════════════════════════════════════════════════════════════════════════
class RLAgent:
    """
    Production DQN agent for drone navigation.
    Supports both inference (async main loop) and training.
    """

    def __init__(
        self,
        logger: logging.Logger,
        model_path: Optional[Path] = None,
        seed: Optional[int] = None,
        state_dim: int = 18,          # ← 18 بدل 16
        action_dim: int = 4,          # ← 4 actions: forward, turn_left, turn_right, hover
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.97,  # ← أسرع: يوصل 0.05 في ~100 episode
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        device: Optional[str] = None,
    ):
        self.logger = logger
        self.model_path = model_path
        self.state_encoder = StateEncoder()
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.train_step_count = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logger.info("RLAgent using device: %s", self.device)

        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayBuffer(capacity=buffer_capacity)

        if self.model_path and self.model_path.exists():
            self.load(self.model_path)
            self.is_dummy = False
            self.logger.info("Loaded trained model from %s", self.model_path)
        else:
            self.logger.warning("No model found at %s. Starting fresh.", self.model_path)
            self.is_dummy = True

    # -------------------------------------------------------------------------
    # Action Selection
    # -------------------------------------------------------------------------
    async def select_action(self, state: Dict[str, Any]) -> str:
        if self.is_dummy:
            action = self._get_random_action()
        elif random.random() < self.epsilon:
            action = self._get_random_action()
        else:
            action = await asyncio.to_thread(self._get_inference_action, state)

        self.logger.debug("RL agent selected action: %s (ε=%.3f)", action, self.epsilon)
        return action

    def _get_random_action(self) -> str:
        """
        Weighted random — forward-heavy (50%) + turns (40%) + hover (10%).
        No up/down — altitude is fixed during navigation training.
        """
        weighted = [
            "forward", "forward", "forward", "forward", "forward",
            "turn_left",  "turn_left",
            "turn_right", "turn_right",
            "hover",
        ]
        return random.choice(weighted)

    def _get_inference_action(self, state: Dict[str, Any]) -> str:
        state_vec = self.state_encoder.encode(state)
        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            action_idx = int(q_values.argmax(dim=1).item())
        return RL_IDX_TO_ACTION[action_idx]

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def remember(
        self,
        state: Dict[str, Any],
        action: str,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        s      = self.state_encoder.encode(state)
        a      = RL_ACTION_TO_IDX[action]
        s_next = self.state_encoder.encode(next_state)
        self.memory.push(s, a, reward, s_next, done)

        if self.is_dummy and len(self.memory) >= self.batch_size:
            self.is_dummy = False
            self.logger.info(
                "✅ Replay buffer ready (%d samples) — switching to RL mode.",
                len(self.memory),
            )

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target_q     = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.logger.info("Target network updated at step %s", self.train_step_count)

        return loss.item()

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def set_eval_mode(self) -> None:
        self.policy_net.eval()
        self.epsilon = 0.0
        if getattr(self, "is_dummy", False):
            self.logger.warning(
                "🚨 RUNNING IN DUMMY MODE: No trained model. Using random actions."
            )
        else:
            self.is_dummy = False

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def save(self, path: Optional[Path] = None) -> None:
        save_path = path or self.model_path or Path("models/dqn_drone_model.pth")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "policy_state":     self.policy_net.state_dict(),
            "target_state":     self.target_net.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "epsilon":          self.epsilon,
            "train_step_count": self.train_step_count,
        }
        torch.save(checkpoint, save_path)
        self.logger.info("Model checkpoint saved to %s", save_path)

    def load(self, path: Path) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.policy_net.load_state_dict(checkpoint["policy_state"])
            self.target_net.load_state_dict(checkpoint["target_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.epsilon          = checkpoint.get("epsilon", 1.0)
            self.train_step_count = checkpoint.get("train_step_count", 0)
            self.policy_net.train()
        except (RuntimeError, KeyError) as e:
            self.logger.warning(
                "⚠️ Model architecture mismatch — starting fresh training. "
                "(Old model had different action_dim.) Error: %s", e
            )
            self.is_dummy = True