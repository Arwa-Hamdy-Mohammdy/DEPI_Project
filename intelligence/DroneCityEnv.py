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
import random
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
        rank: int                        = 0,
    ) -> None:
        super().__init__()

        # ── 1. Store config ────────────────────────────────────────────────
        self._start_voxel      = start
        self._goal_voxel       = goal
        self._max_episode_steps = max_episode_steps
        self.rank              = rank

        # ── 2. Connect to AirSim ──────────────────────────────────────────
        self.vehicle_name = f"Drone{self.rank}"
        logger.info("Connecting to AirSim (Vehicle: %s) …", self.vehicle_name)
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        logger.info("AirSim connection established for %s.", self.vehicle_name)

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
        if self._waypoints and len(self._waypoints) > 1:
            self._waypoints.pop(0)
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
                # center-cropped depth sensor (1,), values in [0, 1] (max 30m)
                "distance_sensor": spaces.Box(
                    low   = 0.0,
                    high  = 1.0,
                    shape = (1,),
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

        # Ensure we are not paused during reset API calls and allow it to register safely
        self.client.simPause(False)
        time.sleep(0.1)

        # -- Safely stop any ongoing motion ----------------------------------
        self.client.cancelLastTask(vehicle_name=self.vehicle_name)
        self.client.armDisarm(False, vehicle_name=self.vehicle_name)

        # -- Full scene reset -------------------------------------------------
        # self.client.reset()  # DEPRECATED: Causes FMallocBinned2 crashes in UE4
        
        # Safe reset: spawn the drone exactly at the start voxel's world coordinate
        # to match the path planner's starting point.
        start_world = self._planner._voxel_to_world(self._start_voxel)
        # start_world is in ENU (x, y, z). AirSim expects NED (x, y, -z)
        start_pose = airsim.Pose(airsim.Vector3r(start_world[0], start_world[1], -start_world[2]), airsim.to_quaternion(0, 0, 0))
        self.client.simSetVehiclePose(start_pose, True, vehicle_name=self.vehicle_name)
        
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        # Allow Unreal Engine physics and rendering to stabilize to prevent crashes
        time.sleep(0.2)

        # We can skip the slow takeoffAsync().join() call entirely since we teleport to air.
        logger.debug("Drone airborne at start voxel altitude.")

        # -- Reset episode bookkeeping ---------------------------------------
        self.current_waypoint_index = 0
        self._step_count            = 0
        self._episode_done          = False
        self._prev_action           = np.zeros(3, dtype=np.float32)

        # -- Randomize Goal --------------------------------------------------
        # Pick a random distance and angle for the new goal to prevent overfitting
        X, Y, _ = self._planner._shape
        start_x, start_y, start_z = self._start_voxel
        
        for _ in range(100):
            radius = random.uniform(20.0, 80.0)
            angle = random.uniform(0.0, 2 * math.pi)
            goal_x = int(start_x + radius * math.cos(angle))
            goal_y = int(start_y + radius * math.sin(angle))
            
            if 0 <= goal_x < X and 0 <= goal_y < Y:
                # Spatial validation check to prevent instant termination
                start_world = self._planner._voxel_to_world((start_x, start_y, start_z))
                goal_world = self._planner._voxel_to_world((goal_x, goal_y, start_z))
                dist_m = math.sqrt((goal_world[0] - start_world[0])**2 + (goal_world[1] - start_world[1])**2)
                
                if dist_m >= 15.0:
                    self._goal_voxel = (goal_x, goal_y, start_z)
                    break
        else:
            # Fallback if no valid point is found (ensure minimum 15m distance safely)
            offset_x = 20 if start_x + 20 < X else -20
            offset_y = 20 if start_y + 20 < Y else -20
            self._goal_voxel = (start_x + offset_x, start_y + offset_y, start_z)

        logger.info("Randomized goal to %s. Re-planning path...", self._goal_voxel)
        self._waypoints = self._planner.plan(self._start_voxel, self._goal_voxel)
        
        # Fallback if no valid path
        if not self._waypoints:
            logger.warning("Planner failed to find path. Falling back to direct line.")
            self._waypoints = [self._start_voxel, self._goal_voxel]
        elif len(self._waypoints) > 1:
            # The first node is always the starting point, remove it
            self._waypoints.pop(0)

        # Remove simPause(True) here to maintain stability with the time.sleep() approach in step()
        
        # Small sleep to throttle RPC calls before fetching state
        time.sleep(0.05)

        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        
        # Exactly initialize the distance tracking using the actual initial state
        pos = state.kinematics_estimated.position
        initial_pos = (pos.x_val, pos.y_val, -pos.z_val)
        self._prev_dist_to_wp = self._dist_to_waypoint(initial_pos, self._waypoints[0])
        
        obs  = self._get_obs(state)
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
            vehicle_name=self.vehicle_name,
        )

        # Replaced simContinueForTime with Python's native sleep to avoid RPC dispatcher crashes
        time.sleep(STEP_DURATION_S)

        # ── 2. Collect new state (Optimised: 1 API call for all kinematics/collisions) 
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        
        self._step_count += 1
        obs       = self._get_obs(state)
        reward, terminated = self._compute_reward(action, state, obs)
        truncated = self._step_count >= self._max_episode_steps

        if terminated or truncated:
            self._episode_done = True
            # Hover to avoid physics instability on next reset
            self.client.hoverAsync(vehicle_name=self.vehicle_name)

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

    def _get_obs(self, state: airsim.MultirotorState) -> Dict[str, np.ndarray]:
        """Assemble the observation dict from live AirSim sensor data.

        Returns
        -------
        dict with keys:
            ``kinematics``     – shape ``(6,)``  float32
            ``waypoint_vector``– shape ``(3,)``  float32
            ``vision``         – shape ``(H,W,1)`` float32  ∈ [0, 1]
            ``distance_sensor``– shape ``(1,)``  float32  ∈ [0, 1]
        """
        # ── Kinematics ────────────────────────────────────────────────────
        kin       = state.kinematics_estimated

        # Position (NED → ENU: flip y and z)
        # Horizontal positions zeroed to prevent overfitting, but altitude (pos_z) is kept
        pos_x =  0.0
        pos_y =  0.0
        pos_z = -kin.position.z_val

        # Linear velocity (same sign-flip for z)
        vel_x =  kin.linear_velocity.x_val
        vel_y =  kin.linear_velocity.y_val
        vel_z = -kin.linear_velocity.z_val

        kinematics_obs = np.array(
            [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z], dtype=np.float32
        )

        # ── Waypoint vector ───────────────────────────────────────────────
        current_pos = (kin.position.x_val, kin.position.y_val, -kin.position.z_val)
        wp          = self._waypoints[self.current_waypoint_index]
        wp_vec      = np.array(
            [wp[0] - current_pos[0], wp[1] - current_pos[1], wp[2] - current_pos[2]], dtype=np.float32
        )

        # ── Vision and Distance ───────────────────────────────────────────
        vision_obs, distance_obs = self._get_depth_vision()

        return {
            "kinematics":      kinematics_obs,
            "waypoint_vector": wp_vec,
            "vision":          vision_obs,
            "distance_sensor": distance_obs,
        }

    def _get_depth_vision(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture and preprocess the depth image and center distance.

        We use a single ImageType.DepthPlanar request to drastically reduce 
        network RPC overhead. DepthPlanar provides the true orthogonal distance 
        to all objects (including buildings), making it perfect for both the 
        CNN vision input and the dedicated center distance sensor.

        Returns
        -------
        vision_obs: np.ndarray shape ``(DEPTH_IMG_H, DEPTH_IMG_W, 1)``  dtype float32.
            Values are normalised to ``[0, 1]`` where 0 = obstacle / minimum
            depth and 1 = maximum depth (``MAX_DEPTH_M``).
        distance_obs: np.ndarray shape ``(1,)`` dtype float32.
            Normalised minimum depth in the center crop [0, 1] where 1 = 30m.
        """
        image_requests = [
            airsim.ImageRequest(
                camera_name = DEPTH_CAM_NAME,
                image_type  = airsim.ImageType.DepthPlanar,
                pixels_as_float = True,
                compress        = False,
            ),
        ]

        responses = self.client.simGetImages(image_requests, vehicle_name=self.vehicle_name)
        depth_response = responses[0]

        if depth_response.width == 0 or depth_response.height == 0:
            logger.warning("Empty depth image received; returning zeros.")
            return np.zeros((DEPTH_IMG_H, DEPTH_IMG_W, 1), dtype=np.float32), np.array([1.0], dtype=np.float32)

        # ── Parse depth image ─────────────────────────────────────────────
        depth_raw = np.array(
            depth_response.image_data_float, dtype=np.float32
        ).reshape(depth_response.height, depth_response.width)

        # Clean NaNs and Infs from depth map to prevent PPO from outputting NaNs
        depth_raw = np.nan_to_num(depth_raw, nan=MAX_DEPTH_M, posinf=MAX_DEPTH_M, neginf=0.0)

        # ── Resize to network input size ──────────────────────────────────
        if depth_raw.shape != (DEPTH_IMG_H, DEPTH_IMG_W):
            depth_raw = self._resize_depth(depth_raw, DEPTH_IMG_H, DEPTH_IMG_W)

        # ── Center Distance Sensor ────────────────────────────────────────
        # Extract a small 20% center crop from the raw un-normalized depth
        ph, pw = depth_raw.shape
        crop_h = max(1, ph // 5)
        crop_w = max(1, pw // 5)
        start_h = (ph - crop_h) // 2
        start_w = (pw - crop_w) // 2
        
        center_crop = depth_raw[start_h : start_h + crop_h, start_w : start_w + crop_w]
        min_depth = float(np.min(center_crop))
        
        # Normalize scalar distance (clip at 30 meters, scale to [0, 1])
        max_dist_m = 30.0
        dist_norm_scalar = np.clip(min_depth, 0.0, max_dist_m) / max_dist_m
        distance_obs = np.array([dist_norm_scalar], dtype=np.float32)

        # ── Clip and normalise vision for CNN [0, 1] ──────────────────────
        depth_norm = np.clip(depth_raw, 0.0, MAX_DEPTH_M) / MAX_DEPTH_M

        return depth_norm[:, :, np.newaxis].astype(np.float32), distance_obs

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
        state: airsim.MultirotorState,
        obs: Dict[str, np.ndarray],
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
        kin      = state.kinematics_estimated
        pos_x    =  kin.position.x_val
        pos_y    =  kin.position.y_val
        pos_z    = -kin.position.z_val    # ENU

        vel_z    = -kin.linear_velocity.z_val   # ENU

        current_pos: Tuple[float, float, float] = (pos_x, pos_y, pos_z)
        wp = self._waypoints[self.current_waypoint_index]

        # ── 1. Collision check (highest priority) ─────────────────────────
        if state.collision.has_collided:
            reward    += COLLISION_REW
            terminated = True
            logger.info(
                "Collision detected at step %d. object=%s",
                self._step_count,
                state.collision.object_name,
            )
            return reward, terminated

        # ── 2. Progress reward (dense) ────────────────────────────────────
        current_dist = self._dist_to_waypoint(current_pos, wp)
        progress     = self._prev_dist_to_wp - current_dist   # positive = closer
        reward      += C1_PROGRESS * progress
        self._prev_dist_to_wp = current_dist

        # ── 3. Time penalty (dense) ───────────────────────────────────────
        reward += TIME_PENALTY

        # ── 3.5. Altitude & Evasion Policies ──────────────────────────────
        # Excessive Ascent Penalty
        if pos_z > 15.0:
            reward -= 0.5  # Strong penalty for flying too high

        # Stuck / Lateral Evasion Reward
        dist_val = float(obs["distance_sensor"][0])
        if dist_val < 0.1:  # Assuming 0.1 is a safe triggering distance (e.g., 4 meters)
            if abs(action[1]) > 0.6:  # Changed from 1.0 to 0.6 to fall within valid action bounds
                reward += 0.5  # Positive reinforcement for dodging laterally
            elif action[0] < 0.5:
                reward -= 1.0  # Penalty for getting stuck without sidestepping
                
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
        kin = self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated
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
            self.client.simPause(False)
            self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
            self.client.armDisarm(False, vehicle_name=self.vehicle_name)
            self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error during graceful shutdown: %s", exc)



