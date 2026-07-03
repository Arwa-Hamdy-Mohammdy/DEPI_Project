"""
DroneCityEnv.py
===============
Gymnasium environment for autonomous drone navigation in AirSim City.

This module bridges three independent subsystems:

1. **AirSim** – physics simulation, sensor data, and flight-control API.
2. **PathPlanner** – pure A* waypoint generator (see ``PathPlanner.py``).
3. **RL Agent (PPO)** – interacts exclusively through the ``gymnasium.Env``
   interface; it never sees AirSim or PathPlanner directly.

Architecture
------------
``DroneCityEnv`` is the *only* class permitted to import from both ``airsim``
and ``PathPlanner``.  All communication is one-directional:

    AirSim ──► DroneCityEnv ──► PathPlanner
    AirSim ◄── DroneCityEnv ◄── RL Agent (gymnasium.Env interface)

Observation Space
-----------------
A ``spaces.Dict`` with three sub-spaces:

- ``kinematics``: shape ``(6,)``  – [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
- ``waypoint_vector``: shape ``(3,)`` – relative (dx, dy, dz) to current waypoint
- ``vision``: shape ``(H, W, 1)`` – mono depth image from the front-facing camera

Action Space
------------
``spaces.Box`` shape ``(3,)`` – continuous velocity command [vx, vy, vz] in m/s,
clipped to ``[-MAX_VEL, MAX_VEL]``.

Usage
-----
>>> env = DroneCityEnv(start=(0,0,5), goal=(40,40,5))
>>> obs, info = env.reset()
>>> obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import airsim
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from intelligence.PathPlanner import PathPlanner, Waypoints

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -- Camera / vision constants -----------------------------------------------
DEPTH_CAM_NAME:  str   = "0"
DEPTH_IMG_H:     int   = 64       # rows (pixels) – resize in AirSim settings
DEPTH_IMG_W:     int   = 64       # columns (pixels)
MAX_DEPTH_M:     float = 50.0     # depth clip value (metres)

# AirSim segmentation colour ID for the "Building" semantic class.
# Set in AirSim's mesh-labelling JSON; default city map uses ID 12 for buildings.
BUILDING_SEG_ID: int   = 12

# -- Flight constants ---------------------------------------------------------
MAX_VEL:             float = 5.0    # m/s  – action space bound
WAYPOINT_RADIUS_M:   float = 2.0   # waypoint is "reached" within this radius
LANDING_VZ_THRESH:   float = 0.5   # |vz| threshold for a safe landing (m/s)
STEP_DURATION_S:     float = 0.1   # seconds of physics per step (joined)

# -- Reward shaping constants ------------------------------------------------
C1_PROGRESS:    float = 2.0    # progress reward scale factor
TIME_PENALTY:   float = -0.05  # applied every step
SMOOTH_PENALTY: float = -0.1   # scale for action smoothness penalty
WP_REWARD:      float = 50.0   # reward for reaching an intermediate waypoint
COLLISION_REW:  float = -100.0 # terminal collision penalty
GOAL_SAFE_REW:  float = 100.0  # terminal reward for smooth final landing
GOAL_CRASH_REW: float = -100.0 # terminal penalty for hard landing

# -- Voxel grid defaults (override at construction if you have a real map) ---
DEFAULT_GRID_SHAPE: Tuple[int, int, int] = (200, 200, 40)
DEFAULT_VOXEL_SIZE: float                = 1.0  # metres


class DroneCityEnv(gym.Env):
    """Autonomous drone navigation environment for AirSim City with PPO.

    Parameters
    ----------
    start:
        Voxel-index start coordinate ``(ix, iy, iz)`` for the A* planner.
    goal:
        Voxel-index goal coordinate ``(ix, iy, iz)`` for the A* planner.
    voxel_grid:
        3-D NumPy uint8 occupancy array.  If ``None`` an empty (all-free)
        grid of shape ``DEFAULT_GRID_SHAPE`` is created.  In production, pass
        a grid generated from AirSim's voxel API or a pre-built map.
    voxel_size:
        Metres per voxel edge for the planner and coordinate conversion.
    max_episode_steps:
        Hard episode cutoff (truncation after this many steps).
    """

    metadata = {"render_modes": []}

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        start: Tuple[int, int, int],
        goal:  Tuple[int, int, int],
        voxel_grid: Optional[np.ndarray] = None,
        voxel_size: float                = DEFAULT_VOXEL_SIZE,
        max_episode_steps: int           = 2_000,
    ) -> None:
        super().__init__()

        # ── 1. Store config ────────────────────────────────────────────────
        self._start_voxel      = start
        self._goal_voxel       = goal
        self._max_episode_steps = max_episode_steps

        # ── 2. Connect to AirSim ──────────────────────────────────────────
        logger.info("Connecting to AirSim …")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        logger.info("AirSim connection established.")

        # ── 3. Build voxel grid and Path Planner ──────────────────────────
        if voxel_grid is None:
            logger.warning(
                "No voxel_grid supplied – using a fully free grid of shape %s. "
                "Pass a real occupancy map for obstacle-aware planning.",
                DEFAULT_GRID_SHAPE,
            )
            voxel_grid = np.zeros(DEFAULT_GRID_SHAPE, dtype=np.uint8)

        self._planner = PathPlanner(
            voxel_grid   = voxel_grid,
            voxel_size   = voxel_size,
            safety_margin = 2,       # keep 2-voxel clearance from buildings
        )

        # ── 4. Pre-compute full global path (waypoints in world-space m) ──
        logger.info("Computing global path …")
        self._waypoints: Waypoints = self._planner.plan(start, goal)
        logger.info("Global path: %d waypoints computed.", len(self._waypoints))

        # ── 5. Episode-level state (initialised properly in reset()) ───────
        self.current_waypoint_index: int            = 0
        self._prev_dist_to_wp:       float          = 0.0
        self._prev_action:           np.ndarray     = np.zeros(3, dtype=np.float32)
        self._step_count:            int            = 0
        self._episode_done:          bool           = False

        # ── 6. Define Gymnasium spaces ────────────────────────────────────
        # Action: continuous velocity command [vx, vy, vz] in m/s
        self.action_space = spaces.Box(
            low   = -MAX_VEL,
            high  =  MAX_VEL,
            shape = (3,),
            dtype = np.float32,
        )

        # Observation dict
        self.observation_space = spaces.Dict(
            {
                # [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
                "kinematics": spaces.Box(
                    low   = -np.inf,
                    high  =  np.inf,
                    shape = (6,),
                    dtype = np.float32,
                ),
                # relative vector from drone to current waypoint [dx, dy, dz]
                "waypoint_vector": spaces.Box(
                    low   = -np.inf,
                    high  =  np.inf,
                    shape = (3,),
                    dtype = np.float32,
                ),
                # mono depth image (H, W, 1), values in [0, 1] after normalisation
                "vision": spaces.Box(
                    low   = 0.0,
                    high  = 1.0,
                    shape = (DEPTH_IMG_H, DEPTH_IMG_W, 1),
                    dtype = np.float32,
                ),
            }
        )

        logger.info("DroneCityEnv initialised successfully.")

    # -----------------------------------------------------------------------
    # gymnasium.Env interface
    # -----------------------------------------------------------------------

    def reset(
        self,
        *,
        seed:    Optional[int]          = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the simulation to the start state.

        Disarms, resets the AirSim scene, re-arms, takes off, then returns
        the first observation.

        Parameters
        ----------
        seed:
            Random seed forwarded to the parent class (for reproducibility).
        options:
            Reserved for future use (ignored).

        Returns
        -------
        observation:
            Initial observation dict matching ``self.observation_space``.
        info:
            Auxiliary diagnostic information.
        """
        super().reset(seed=seed)

        logger.debug("Resetting AirSim environment …")

        # -- Safely stop any ongoing motion ----------------------------------
        self.client.cancelLastTask()
        self.client.armDisarm(False)

        # -- Full scene reset -------------------------------------------------
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # -- Take off to a safe altitude -------------------------------------
        self.client.takeoffAsync().join()
        logger.debug("Drone airborne.")

        # -- Reset episode bookkeeping ---------------------------------------
        self.current_waypoint_index = 0
        self._step_count            = 0
        self._episode_done          = False
        self._prev_action           = np.zeros(3, dtype=np.float32)

        # Compute initial distance to first waypoint
        initial_pos = self._get_position()
        self._prev_dist_to_wp = self._dist_to_waypoint(
            initial_pos, self._waypoints[0]
        )

        obs  = self._get_obs()
        info = {"waypoint_index": self.current_waypoint_index,
                "num_waypoints":  len(self._waypoints)}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Sends a velocity command to AirSim, advances the simulation by
        ``STEP_DURATION_S`` seconds, then computes the next observation and
        reward.

        Parameters
        ----------
        action:
            NumPy array of shape ``(3,)`` – desired [vx, vy, vz] in m/s.

        Returns
        -------
        observation:   Next observation dict.
        reward:        Scalar reward for this step.
        terminated:    ``True`` if the episode ended due to a game event
                       (collision or goal reached).
        truncated:     ``True`` if ``max_episode_steps`` was exceeded.
        info:          Diagnostic dictionary.
        """
        assert not self._episode_done, (
            "step() called after the episode has ended. Call reset() first."
        )

        # ── 1. Clip action and send velocity command ───────────────────────
        action = np.clip(action, -MAX_VEL, MAX_VEL).astype(np.float32)
        vx, vy, vz = float(action[0]), float(action[1]), float(action[2])

        # AirSim NED convention: vz is inverted (positive = downward in NED).
        # We expose vz in ENU (positive = upward) to the agent for intuition.
        self.client.moveByVelocityAsync(
            vx    = vx,
            vy    = vy,
            vz    = -vz,          # ENU → NED sign flip
            duration = STEP_DURATION_S,
            drivetrain  = airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode    = airsim.YawMode(is_rate=False, yaw_or_rate=0),
        ).join()

        # ── 2. Collect new state ──────────────────────────────────────────
        self._step_count += 1
        obs       = self._get_obs()
        reward, terminated = self._compute_reward(action)
        truncated = self._step_count >= self._max_episode_steps

        if terminated or truncated:
            self._episode_done = True
            # Hover to avoid physics instability on next reset
            self.client.hoverAsync()

        # Update previous action for smoothness penalty in the next step
        self._prev_action = action.copy()

        info: Dict[str, Any] = {
            "step":             self._step_count,
            "waypoint_index":   self.current_waypoint_index,
            "num_waypoints":    len(self._waypoints),
            "terminated":       terminated,
            "truncated":        truncated,
        }

        logger.debug(
            "step=%d  wp=%d/%d  reward=%.3f  term=%s  trunc=%s",
            self._step_count,
            self.current_waypoint_index,
            len(self._waypoints),
            reward,
            terminated,
            truncated,
        )

        return obs, reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Assemble the observation dict from live AirSim sensor data.

        Returns
        -------
        dict with keys:
            ``kinematics``     – shape ``(6,)``  float32
            ``waypoint_vector``– shape ``(3,)``  float32
            ``vision``         – shape ``(H,W,1)`` float32  ∈ [0, 1]
        """
        # ── Kinematics ────────────────────────────────────────────────────
        state     = self.client.getMultirotorState()
        kin       = state.kinematics_estimated

        # Position (NED → ENU: flip y and z)
        pos_x =  kin.position.x_val
        pos_y =  kin.position.y_val
        pos_z = -kin.position.z_val   # NED z is downward; flip to ENU

        # Linear velocity (same sign-flip for z)
        vel_x =  kin.linear_velocity.x_val
        vel_y =  kin.linear_velocity.y_val
        vel_z = -kin.linear_velocity.z_val

        kinematics_obs = np.array(
            [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z], dtype=np.float32
        )

        # ── Waypoint vector ───────────────────────────────────────────────
        current_pos = (pos_x, pos_y, pos_z)
        wp          = self._waypoints[self.current_waypoint_index]
        wp_vec      = np.array(
            [wp[0] - pos_x, wp[1] - pos_y, wp[2] - pos_z], dtype=np.float32
        )

        # Update distance tracker for reward computation
        self._prev_dist_to_wp = self._dist_to_waypoint(current_pos, wp)

        # ── Vision (depth + building segmentation) ────────────────────────
        vision_obs = self._get_depth_vision()

        return {
            "kinematics":      kinematics_obs,
            "waypoint_vector": wp_vec,
            "vision":          vision_obs,
        }

    def _get_depth_vision(self) -> np.ndarray:
        """Capture and preprocess the depth image with building-class masking.

        Retrieves both a depth perspective image and a segmentation image from
        AirSim. Pixels classified as **Building** (segmentation ID
        ``BUILDING_SEG_ID``) are set to zero depth, emphasising these obstacles
        in the agent's visual field.

        Returns
        -------
        np.ndarray  shape ``(DEPTH_IMG_H, DEPTH_IMG_W, 1)``  dtype float32.
            Values are normalised to ``[0, 1]`` where 0 = obstacle / minimum
            depth and 1 = maximum depth (``MAX_DEPTH_M``).
        """
        image_requests = [
            airsim.ImageRequest(
                camera_name = DEPTH_CAM_NAME,
                image_type  = airsim.ImageType.DepthPerspective,
                pixels_as_float = True,
                compress        = False,
            ),
            airsim.ImageRequest(
                camera_name = DEPTH_CAM_NAME,
                image_type  = airsim.ImageType.Segmentation,
                pixels_as_float = False,
                compress        = False,
            ),
        ]

        responses = self.client.simGetImages(image_requests)

        # ── Parse depth image ─────────────────────────────────────────────
        depth_response = responses[0]
        if depth_response.width == 0 or depth_response.height == 0:
            logger.warning("Empty depth image received; returning zeros.")
            return np.zeros((DEPTH_IMG_H, DEPTH_IMG_W, 1), dtype=np.float32)

        depth_raw = np.array(
            depth_response.image_data_float, dtype=np.float32
        ).reshape(depth_response.height, depth_response.width)

# ── Parse segmentation image ──────────────────────────────────────
        seg_response = responses[1]
        if seg_response.width > 0 and seg_response.height > 0:
            seg_raw = np.frombuffer(seg_response.image_data_uint8, dtype=np.uint8)
            seg_raw = seg_raw.reshape(
                seg_response.height, seg_response.width, -1
            )
            
            # استخراج القناة الحمراء (اللي فيها الـ ID بتاع المباني)
            seg_channel = seg_raw[:, :, 0]
            
            # --- التعديل هنا: توحيد المقاسات لتطابق صورة العمق ---
            if seg_channel.shape != depth_raw.shape:
                seg_channel = self._resize_depth(seg_channel, depth_raw.shape[0], depth_raw.shape[1])
            
            # تحديد المباني وتعديل العمق
            building_mask = seg_channel == BUILDING_SEG_ID
            depth_raw[building_mask] = 0.0
        else:
            logger.debug("Empty segmentation image; skipping building masking.")

        # ── Resize to network input size ──────────────────────────────────
        # Use simple area interpolation via strided slicing if shapes match,
        # otherwise fall back to nearest-neighbour via index arithmetic.
        if depth_raw.shape != (DEPTH_IMG_H, DEPTH_IMG_W):
            depth_raw = self._resize_depth(depth_raw, DEPTH_IMG_H, DEPTH_IMG_W)

        # ── Clip and normalise to [0, 1] ──────────────────────────────────
        depth_norm = np.clip(depth_raw, 0.0, MAX_DEPTH_M) / MAX_DEPTH_M

        return depth_norm[:, :, np.newaxis].astype(np.float32)

    @staticmethod
    def _resize_depth(
        img: np.ndarray, out_h: int, out_w: int
    ) -> np.ndarray:
        """Nearest-neighbour downsample / upsample with no extra dependencies."""
        src_h, src_w = img.shape
        row_idx = (np.arange(out_h) * src_h / out_h).astype(int)
        col_idx = (np.arange(out_w) * src_w / out_w).astype(int)
        return img[np.ix_(row_idx, col_idx)]

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------

    def _compute_reward(
        self,
        action: np.ndarray,
    ) -> Tuple[float, bool]:
        """Compute the scalar reward for the current step.

        Reward components
        -----------------
        1. **Progress reward** (dense): proportional to reduction in distance
           to the current waypoint.
        2. **Time penalty** (dense): small constant negative reward every step
           to encourage efficiency.
        3. **Smoothness penalty** (dense): penalises large changes between
           consecutive action vectors to promote smooth flight.
        4. **Waypoint reached** (sparse): large bonus when within
           ``WAYPOINT_RADIUS_M`` of the current intermediate waypoint.
        5. **Collision** (terminal, negative): episode ends with a large penalty
           if the drone contacts any object.
        6. **Goal reached** (terminal): if the *final* waypoint is reached,
           the agent is rewarded for a smooth landing (|vz| < threshold) or
           penalised for a hard impact.

        Parameters
        ----------
        action:
            Current action vector, used to compute the smoothness penalty.

        Returns
        -------
        reward:      Total reward for this step.
        terminated:  ``True`` if the episode should end.
        """
        terminated = False
        reward     = 0.0

        # ── Fetch live state ──────────────────────────────────────────────
        state    = self.client.getMultirotorState()
        kin      = state.kinematics_estimated
        pos_x    =  kin.position.x_val
        pos_y    =  kin.position.y_val
        pos_z    = -kin.position.z_val    # ENU

        vel_z    = -kin.linear_velocity.z_val   # ENU

        current_pos: Tuple[float, float, float] = (pos_x, pos_y, pos_z)
        wp = self._waypoints[self.current_waypoint_index]

        # ── 1. Collision check (highest priority) ─────────────────────────
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            reward    += COLLISION_REW
            terminated = True
            logger.info(
                "Collision detected at step %d. object=%s",
                self._step_count,
                collision_info.object_name,
            )
            return reward, terminated

        # ── 2. Progress reward (dense) ────────────────────────────────────
        current_dist = self._dist_to_waypoint(current_pos, wp)
        progress     = self._prev_dist_to_wp - current_dist   # positive = closer
        reward      += C1_PROGRESS * progress
        self._prev_dist_to_wp = current_dist

        # ── 3. Time penalty (dense) ───────────────────────────────────────
        reward += TIME_PENALTY

        # ── 4. Smoothness penalty (dense) ─────────────────────────────────
        action_delta  = np.linalg.norm(action - self._prev_action)
        reward       += SMOOTH_PENALTY * action_delta

        # ── 5. Waypoint reached (sparse) ─────────────────────────────────
        if self._is_waypoint_reached(current_pos, wp):
            is_final_wp = (
                self.current_waypoint_index == len(self._waypoints) - 1
            )

            if is_final_wp:
                # ── 6. Final waypoint / landing check (terminal) ──────────
                if abs(vel_z) < LANDING_VZ_THRESH:
                    reward    += GOAL_SAFE_REW
                    logger.info(
                        "Goal reached with safe landing at step %d (|vz|=%.2f).",
                        self._step_count, abs(vel_z),
                    )
                else:
                    reward    += GOAL_CRASH_REW
                    logger.info(
                        "Goal reached but CRASH LANDING at step %d (|vz|=%.2f).",
                        self._step_count, abs(vel_z),
                    )
                terminated = True
            else:
                # Intermediate waypoint reached
                reward += WP_REWARD
                self.current_waypoint_index += 1
                logger.info(
                    "Waypoint %d/%d reached at step %d.",
                    self.current_waypoint_index,
                    len(self._waypoints),
                    self._step_count,
                )
                # Recalculate distance to the newly activated waypoint
                new_wp = self._waypoints[self.current_waypoint_index]
                self._prev_dist_to_wp = self._dist_to_waypoint(
                    current_pos, new_wp
                )

        return reward, terminated

    # -----------------------------------------------------------------------
    # Helper utilities
    # -----------------------------------------------------------------------

    def _get_position(self) -> Tuple[float, float, float]:
        """Return the current drone position in ENU world-space metres."""
        kin = self.client.getMultirotorState().kinematics_estimated
        return (
             kin.position.x_val,
             kin.position.y_val,
            -kin.position.z_val,  # NED z → ENU z
        )

    @staticmethod
    def _dist_to_waypoint(
        pos: Tuple[float, float, float],
        wp:  Tuple[float, float, float],
    ) -> float:
        """Euclidean distance in 3-D between *pos* and waypoint *wp*."""
        return math.sqrt(
            (wp[0] - pos[0]) ** 2
            + (wp[1] - pos[1]) ** 2
            + (wp[2] - pos[2]) ** 2
        )

    @staticmethod
    def _is_waypoint_reached(
        pos: Tuple[float, float, float],
        wp:  Tuple[float, float, float],
        radius: float = WAYPOINT_RADIUS_M,
    ) -> bool:
        """Return ``True`` if the drone is within *radius* metres of *wp*."""
        return DroneCityEnv._dist_to_waypoint(pos, wp) <= radius

    def close(self) -> None:
        """Cleanly disconnect from AirSim.

        Always call this when you are finished with the environment to avoid
        leaving the simulation in an armed / API-controlled state.
        """
        logger.info("Closing DroneCityEnv – releasing AirSim control.")
        try:
            self.client.hoverAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error during graceful shutdown: %s", exc)



