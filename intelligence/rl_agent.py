"""
rl_agent.py
===========
Async inference wrapper around a Stable-Baselines3 PPO policy trained with
``DroneCityEnv`` (see DroneCityEnv.py — that file is the ground truth for the
observation/action contract this class is bound to).

This replaces the previous Dueling-DQN agent entirely. There is no replay
buffer, no epsilon-greedy exploration, and no hand-rolled state encoder here
on purpose: PPO is off-policy-free at inference time, and the Dict
observation is built directly by ``main.py._build_consistent_state()`` to
match ``DroneCityEnv.observation_space``. This class's only job is

    obs (Dict[str, np.ndarray])  ──►  PPO.predict()  ──►  action (np.ndarray)

Everything else (kinematics fetching, vision preprocessing, sending the
action to AirSim) lives outside this class, by design — it has no AirSim
dependency at all, so it can be unit-tested with plain NumPy fixtures.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from stable_baselines3 import PPO

# Must match DroneCityEnv.action_space exactly: Box(-MAX_VEL, MAX_VEL, shape=(3,))
ACTION_DIM: int = 3


class RLAgent:
    """Production wrapper for SB3 PPO inference inside the async control loop."""

    def __init__(
        self,
        logger: logging.Logger,
        model_path: Path,
        device: str = "cpu",
    ) -> None:
        self.logger = logger
        self.model_path = Path(model_path)
        self._deterministic = True  # PPO inference defaults to deterministic actions

        if not self.model_path.exists():
            # Deliberately NOT falling back to a "dummy" random-action mode
            # the way the old DQN agent did. A PPO policy commands continuous
            # velocity directly — flying on random velocity vectors is a
            # safety hazard, not a graceful degradation. Fail loudly, and do
            # it before the drone ever arms (see the pre-flight check in
            # main.py.run_control_loop).
            raise FileNotFoundError(
                f"PPO model not found at '{self.model_path}'. Refusing to "
                "start a flight without a trained policy."
            )

        self.logger.info("Loading PPO policy from %s ...", self.model_path)
        self.model: PPO = PPO.load(str(self.model_path), device=device)
        self.logger.info(
            "PPO policy loaded | device=%s | obs_space=%s | action_space=%s",
            device, self.model.observation_space, self.model.action_space,
        )

        if tuple(self.model.action_space.shape) != (ACTION_DIM,):
            raise ValueError(
                f"Loaded policy action_space={self.model.action_space} does not "
                f"match the expected shape ({ACTION_DIM},). Wrong checkpoint?"
            )

    # -------------------------------------------------------------------------
    # Mode toggles — kept for interface parity with the legacy DQN agent so
    # main.py's `agent.set_eval_mode()` call keeps working unchanged.
    # -------------------------------------------------------------------------
    def set_eval_mode(self) -> None:
        """PPO inference is deterministic by default; this documents intent
        and is safe to call every time, same as before."""
        self._deterministic = True
        self.logger.info("RLAgent set to EVAL mode (deterministic PPO actions).")

    def set_stochastic_mode(self) -> None:
        """Sample from the action distribution instead of taking its mean.
        Useful for domain-randomization / robustness testing in sim only —
        do not use this for real flight."""
        self._deterministic = False
        self.logger.warning("RLAgent set to STOCHASTIC mode — actions will vary run-to-run.")

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    async def select_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run one PPO forward pass off the event loop.

        Parameters
        ----------
        obs:
            Dict with keys "kinematics" (6,), "waypoint_vector" (3,),
            "vision" (64,64,1) — must match DroneCityEnv.observation_space
            exactly in shape and dtype. Build it with
            ``main.py._build_consistent_state()``.

        Returns
        -------
        np.ndarray shape (3,), dtype float32 — [vx, vy, vz] in m/s. SB3
        already clips this to [-MAX_VEL, MAX_VEL] internally (the action
        space bounds), so no extra clipping is required here.
        """
        try:
            action = await asyncio.to_thread(self._predict_sync, obs)
            self.logger.debug(
                "PPO action: vx=%.2f vy=%.2f vz=%.2f", action[0], action[1], action[2]
            )
            return action
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.error("PPO inference failed (%s) — defaulting to hover.", exc)
            return np.zeros(ACTION_DIM, dtype=np.float32)

    def _predict_sync(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        action, _next_state = self.model.predict(obs, deterministic=self._deterministic)
        return action.astype(np.float32)
