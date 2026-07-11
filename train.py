"""
train.py
========
PPO Training Script for Autonomous Drone Navigation in AirSim City.

This script wires together:
  - DroneCityEnv  (gymnasium environment with AirSim + PathPlanner)
  - stable_baselines3 PPO  (learning algorithm)
  - NatureCNN  (convolutional feature extractor for depth image observations)
  - Monitor + EvalCallback  (logging and best-model tracking)

Run
---
    python train.py                          # fresh training
    python train.py --resume ./models/checkpoints/ppo_drone_10000_steps.zip
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from intelligence.DroneCityEnv import DroneCityEnv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters (all explicit — no hidden defaults)
# ---------------------------------------------------------------------------

# -- Environment --
START_VOXEL  = (0,  0,  5)
GOAL_VOXEL   = (40, 40, 5)
SEED         = 42           # fixed seed for reproducibility

# -- PPO core --
TOTAL_TIMESTEPS = 500_000  # 500k is too low for 3-D drone navigation
N_STEPS      = 2048    # rollout length per environment update
BATCH_SIZE   = 256     # minibatch size (2048/256 = 8 minibatches per epoch)
N_EPOCHS     = 10      # gradient passes over each rollout buffer
GAMMA        = 0.99    # discount factor
GAE_LAMBDA   = 0.95    # GAE λ — bias/variance trade-off for advantage estimates
LEARNING_RATE = 3e-4   # Adam learning rate

# -- PPO loss coefficients (must match RL_Agent_Mathematical_Models.md §9.4) --
CLIP_RANGE   = 0.2     # ε — trust-region clipping threshold
ENT_COEF     = 0.01    # c₂ — entropy bonus (CRITICAL: keeps exploration alive)
VF_COEF      = 0.5     # c₁ — value function loss weight
MAX_GRAD_NORM = 0.5    # gradient clipping norm (prevents exploding gradients)

# -- Evaluation & checkpointing --
EVAL_FREQ    = 20_000  # evaluate every N environment steps
N_EVAL_EPS   = 5       # number of episodes per evaluation run
SAVE_FREQ    = 10_000  # checkpoint every N steps

# -- Paths --
LOG_DIR      = "./data/tensorboard_logs/"
SAVE_DIR     = "./models/checkpoints/"
EVAL_DIR     = "./models/best_model/"
FINAL_PATH   = "./models/ppo_drone_final"


# ---------------------------------------------------------------------------
# Policy network configuration
# ---------------------------------------------------------------------------

def build_policy_kwargs() -> dict:
    """
    Define the neural network architecture for MultiInputPolicy.

    DroneCityEnv uses a Dict observation space:
      - 'vision'          : (64, 64, 1)  depth image  → processed by NatureCNN
      - 'kinematics'      : (6,)         pos + vel    → processed by MLP
      - 'waypoint_vector' : (3,)         rel. waypoint→ processed by MLP

    SB3's MultiInputPolicy automatically routes image keys through a CNN and
    vector keys through a small MLP, then concatenates the features.
    We override the CNN to use NatureCNN (the architecture from DQN / Atari)
    and set a deeper shared MLP to handle the fused representation.

    Returns
    -------
    dict  passed directly to PPO(policy_kwargs=...)
    """

    return dict(
        features_extractor_class=DroneCustomExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        # تعديل الهيكل ليناسب SB3 v1.8.0 (استخدام dict بدلاً من list)
        net_arch=dict(pi=[256, 128], vf=[256, 128])
    )


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(rank: int, seed: int = SEED):
    """
    Instantiate and wrap DroneCityEnv for training.

    Wrapping order matters:
      DroneCityEnv  →  Monitor
    Monitor must be the outermost wrapper so SB3 can read episode stats.
    """
    def _init() -> Monitor:
        # Offset the start voxel based on rank to prevent UE4 physical collision crashes
        # e.g., Drone0 at (0, 0, 5), Drone1 at (2, 2, 5), Drone2 at (4, 4, 5)
        offset_start = (
            START_VOXEL[0] + rank * 2,
            START_VOXEL[1] + rank * 2,
            START_VOXEL[2]
        )
        env = DroneCityEnv(start=offset_start, goal=GOAL_VOXEL, rank=rank)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DroneCustomExtractor(BaseFeaturesExtractor):
    """
    شبكة عصبية مخصصة لمعالجة الـ State المركبة للدرون.
    تفصل معالجة صور الكاميرا عن معالجة الأرقام (السرعة والمسافات).
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # استدعاء الكلاس الأساسي
        super().__init__(observation_space, features_dim)
        
        extractors = {}
        total_concat_size = 0
        
        for key, subspace in observation_space.spaces.items():
            if key == "vision":
                # 1. معالجة صورة الكاميرا (Depth) باستخدام شبكة CNN
                # SB3 بيعدل أبعاد الصورة أوتوماتيكياً لتكون (Channels, Height, Width)
                n_input_channels = subspace.shape[2]
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Flatten()
                )
                
                # حساب حجم المخرجات من الـ CNN رياضياً
                with th.no_grad():
                    # محاكاة مرور صورة وهمية لحساب الأبعاد
                    sample_img = th.as_tensor(subspace.sample()[None]).float()
                    sample_img = sample_img.permute(0, 3, 1, 2)  # HWC to CHW
                    cnn_out_dim = cnn(sample_img).shape[1]
                
                extractors[key] = cnn
                total_concat_size += cnn_out_dim
                
            elif key in ["kinematics", "waypoint_vector", "distance_sensor"]:
                # 2. معالجة الأرقام (السرعة والمسافة) باستخدام Linear Layers
                linear = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64),
                    nn.ReLU()
                )
                extractors[key] = linear
                total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors)
        
        # 3. الطبقة النهائية لدمج مخرجات الكاميرا مع الأرقام
        self._features_dim = features_dim
        self.final_net = nn.Sequential(
            nn.Linear(total_concat_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> th.Tensor:  # <-- لازم تكون هنا بالظبط!
            encoded_tensor_list = []
            
            for key, extractor in self.extractors.items():
                if key == "vision":
                    obs_vision = observations[key].permute(0, 3, 1, 2)
                    encoded_tensor_list.append(extractor(obs_vision))
                else:
                    encoded_tensor_list.append(extractor(observations[key]))
            
            concat_features = th.cat(encoded_tensor_list, dim=1)
            return self.final_net(concat_features)

# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main(resume_path: Optional[str] = None) -> None:
    """
    Run the PPO training loop.

    Parameters
    ----------
    resume_path:
        Path to a saved .zip checkpoint to resume from.
        If None, training starts fresh.
    """
    logger.info("Starting PPO Training for Drone Navigation in AirSim City.")
    logger.info("Device: %s", "cuda" if torch.cuda.is_available() else "cpu (WARN: slow)")

    # -- 1. Reproducibility ---------------------------------------------------
    set_random_seed(SEED)

    # -- 2. Create directories ------------------------------------------------
    for directory in (LOG_DIR, SAVE_DIR, EVAL_DIR):
        os.makedirs(directory, exist_ok=True)

    # -- 3. Training environment ----------------------------------------------
    num_envs = 5
    train_env = SubprocVecEnv([make_env(i, seed=SEED) for i in range(num_envs)])

    # -- 4. Evaluation environment (separate instance — never used for training)
    # EvalCallback needs its own env so evaluation episodes don't corrupt
    # the training rollout buffer.
    # Removed eval_env to prevent AirSim multi-instance sync crashes.
    # eval_env = DummyVecEnv([make_env(num_envs, seed=SEED + 1)])

    # -- 5. Build or load PPO model ------------------------------------------
    if resume_path:
        logger.info("Resuming training from checkpoint: %s", resume_path)
        from stable_baselines3.common.utils import get_schedule_fn
        custom_objects = {
            "lr_schedule": get_schedule_fn(LEARNING_RATE),
            "clip_range": get_schedule_fn(CLIP_RANGE),
        }
        model = PPO.load(
            resume_path,
            env=train_env,
            device="cuda",        # respects CUDA if available
            custom_objects=custom_objects
            # Do NOT reset timestep counter so TensorBoard continues the curve
        )
    else:
        logger.info("Initialising fresh PPO model.")
        model = PPO(
            policy           = "MultiInputPolicy",  # handles Dict obs (images + vectors)
            env              = train_env,

            # --- Rollout collection ---
            n_steps          = N_STEPS,       # steps per env before each update
            gamma            = GAMMA,         # discount factor γ
            gae_lambda       = GAE_LAMBDA,    # GAE λ (bias/variance trade-off)

            # --- PPO update ---
            learning_rate    = LEARNING_RATE,
            n_epochs         = N_EPOCHS,      # gradient passes over each rollout
            batch_size       = BATCH_SIZE,    # minibatch size
            clip_range       = CLIP_RANGE,    # ε — trust-region clip threshold

            # --- Loss coefficients (L^PPO = L^CLIP - c1*L^VF + c2*H) ---
            vf_coef          = VF_COEF,       # c₁ = 0.5
            ent_coef         = ENT_COEF,      # c₂ = 0.01 — keeps exploration alive

            # --- Gradient stability ---
            max_grad_norm    = MAX_GRAD_NORM, # clip gradient norm

            # --- Network architecture ---
            policy_kwargs    = build_policy_kwargs(),

            # --- Logging ---
            tensorboard_log  = LOG_DIR,
            verbose          = 1,
            seed             = SEED,
            device           = "cuda",        # auto: uses CUDA if available
        )

    logger.info("Model policy:\n%s", model.policy)

    # -- 6. Callbacks ---------------------------------------------------------

    # 6a. Save a checkpoint every SAVE_FREQ environment steps
    checkpoint_cb = CheckpointCallback(
        save_freq   = SAVE_FREQ,
        save_path   = SAVE_DIR,
        name_prefix = "ppo_drone",
        verbose     = 1,
    )

    # 6b. Evaluate periodically and save the BEST model separately.
    #     This is the model you should deploy — not necessarily the final one.
    # Disabled EvalCallback to prevent AirSim multi-instance sync crashes.
    # eval_cb = EvalCallback( ... )

    callbacks = CallbackList([checkpoint_cb])

    # -- 7. Training ----------------------------------------------------------
    try:
        model.learn(
            total_timesteps     = TOTAL_TIMESTEPS,
            callback            = callbacks,
            tb_log_name         = "PPO_CityEnv_Run1",
            reset_num_timesteps = resume_path is None,  # False when resuming
            progress_bar        = True,
        )
        logger.info("Training completed successfully.")

    except KeyboardInterrupt:
        logger.warning("Training interrupted manually by user (Ctrl+C).")

    except Exception as exc:
        # Catch AirSim crashes, OOM errors, etc. — always save before dying
        logger.error("Unexpected error during training: %s", exc, exc_info=True)

    finally:
        # Always save the current model and cleanly close AirSim
        logger.info("Saving final model to: %s", FINAL_PATH)
        model.save(FINAL_PATH)
        train_env.close()
        # eval_env.close()
        logger.info("Training session ended. Model saved.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO drone navigation agent in AirSim City."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="./models/checkpoints/ppo_drone_330000_steps.zip",
        metavar="CHECKPOINT_PATH",
        help="Path to a .zip checkpoint to resume training from.",
    )
    args = parser.parse_args()
    main(resume_path=args.resume)