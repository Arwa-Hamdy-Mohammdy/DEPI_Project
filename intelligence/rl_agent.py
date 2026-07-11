import asyncio
import logging
from typing import Any, Dict, Optional
from pathlib import Path
import numpy as np

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


class RLAgent:
    """
    Inference-only agent for drone navigation using a trained Stable Baselines3 PPO model.
    """

    def __init__(
        self,
        logger: logging.Logger,
        model_path: Optional[Path] = None,
    ):
        self.logger = logger
        self.model_path = model_path
        self.model = None

        if PPO is None:
            self.logger.error("stable_baselines3 is not installed. PPO model cannot be loaded.")
            self.is_dummy = True
            return

        if self.model_path and self.model_path.exists():
            try:
                # Load the PPO model using Stable Baselines3
                self.model = PPO.load(str(self.model_path))
                self.is_dummy = False
                self.logger.info("Loaded trained PPO model from %s", self.model_path)
            except Exception as e:
                self.logger.error("Failed to load PPO model from %s: %s", self.model_path, e)
                self.is_dummy = True
        else:
            self.logger.warning("No model found at %s. Running in dummy mode (zero actions).", self.model_path)
            self.is_dummy = True

    # -------------------------------------------------------------------------
    # Action Selection
    # -------------------------------------------------------------------------
    async def select_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Takes raw dictionary observation and returns continuous actions [vx, vy, vz].
        """
        if self.is_dummy or self.model is None:
            action = self._get_dummy_action()
        else:
            # Run inference in a separate thread to avoid blocking the asyncio event loop
            action = await asyncio.to_thread(self._get_inference_action, state)

        self.logger.debug("RL agent selected action: %s", action)
        return action

    def _get_dummy_action(self) -> np.ndarray:
        """
        Fallback action if no model is loaded. Returns zero velocities (hover).
        """
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _get_inference_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Synchronous method to run model inference.
        """
        # The model expects the raw observation dictionary natively via MultiInputPolicy.
        # SB3 predict returns a tuple: (action, state). We only need the action.
        action, _ = self.model.predict(state, deterministic=True)
        return action

    # -------------------------------------------------------------------------
    # Interface Compatibility
    # -------------------------------------------------------------------------
    def set_eval_mode(self) -> None:
        """
        Provided for compatibility if the main loop still calls it.
        SB3 models are inherently used in eval mode when calling predict.
        """
        pass