from __future__ import annotations

import asyncio
import math
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

class DroneController:
    """Async wrapper around AirSim with a single dedicated thread for all RPC."""

    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.connected = False
        self.last_connect_error: str = ""
        # CRITICAL: max_workers=1 guarantees the AirSim client is never accessed
        # concurrently, eliminating msgpack BufferError and Tornado IOLoop races.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="airsim_ctl")

    # -------------------------------------------------------------------------
    # Internal blocking helpers (run exclusively inside self._executor)
    # -------------------------------------------------------------------------
    def _connect_sync(self, ip: str = "127.0.0.1", port: int = 41451):
        import airsim  # type: ignore
        self.logger.info(f"Attempting AirSim RPC connection at {ip}:{port}...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((ip, port))
            sock.close()
        except OSError as exc:
            raise ConnectionError(
                f"AirSim RPC port not reachable at {ip}:{port}. "
                "Verify the simulator is running and the port is correct."
            ) from exc

        client = airsim.MultirotorClient(ip=ip, port=port)
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        return client

    def _takeoff_sync(self, client):
        client.takeoffAsync().join()

    def _land_sync(self, client):
        client.landAsync().join()

    def _move_sync(self, client, vx, vy, vz, duration):
        import airsim
        if vz == 0.0:
            # Fix: Hold altitude strictly when not intentionally moving up/down
            state = client.getMultirotorState()
            z = state.kinematics_estimated.position.z_val
            client.moveByVelocityZAsync(vx, vy, z, duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0))
        else:
            client.moveByVelocityAsync(vx, vy, vz, duration)

    def _ascend_sync(self, client, z, velocity):
        client.moveToZAsync(z, velocity).join()

    def _get_pose_sync(self, client):
        state = client.getMultirotorState()
        return {
            "x": state.kinematics_estimated.position.x_val,
            "y": state.kinematics_estimated.position.y_val,
            "z": state.kinematics_estimated.position.z_val,
        }

    def _hover_sync(self, client):
        import airsim
        state = client.getMultirotorState()
        z = state.kinematics_estimated.position.z_val
        client.moveByVelocityZAsync(0.0, 0.0, z, 1.0, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join()

    def _has_collided_sync(self, client):
        info = client.simGetCollisionInfo()
        if info.has_collided:
            if (
                "Road" in info.object_name
                or "Landscape" in info.object_name
                or "Ground" in info.object_name
                or "Floor" in info.object_name
                or info.object_name == ""
            ):
                return False
            self.logger.warning(
                "Collision detected! Object: %s, Penetration: %s",
                info.object_name, info.penetration_depth,
            )
        return info.has_collided

    def _disconnect_sync(self, client):
        client.armDisarm(False)
        client.enableApiControl(False)

    def _reset_sync(self, client):
        # client.reset() # DEPRECATED: Causes FMallocBinned2 crashes in UE4
        import airsim
        start_pose = airsim.Pose(airsim.Vector3r(0, 0, -5), airsim.to_quaternion(0, 0, 0))
        client.simSetVehiclePose(start_pose, True)
        client.enableApiControl(True)
        client.armDisarm(True)

    def _print_to_game_sync(self, client, title, message):
        client.simPrintLogMessage(title, message, 2)

    # ── NEW: Yaw helpers ──────────────────────────────────────────────────────
    def _get_yaw_sync(self, client) -> float:
        """
        Returns the current yaw of the drone in degrees [-180, 180].
        Uses the AirSim quaternion from the estimated kinematics.
        """
        state = client.getMultirotorState()
        q = state.kinematics_estimated.orientation
        # Quaternion → yaw (degrees).  AirSim quaternion fields: w_val, x_val, y_val, z_val
        siny_cosp = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        return math.degrees(yaw_rad)

    def _rotate_yaw_sync(self, client, degrees: float, duration: float = 1.5):
        import airsim
        state = client.getMultirotorState()
        current_z = state.kinematics_estimated.position.z_val
        yaw_rate = degrees / duration          # degrees/sec
        
        # Fix: Lock vx=0, vy=0 to prevent drifting during rotation!
        client.moveByVelocityZAsync(
            0.0, 0.0, current_z, duration,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, yaw_rate)
        ).join()

    # -------------------------------------------------------------------------
    # Public async API
    # -------------------------------------------------------------------------
    async def connect(self, timeout: int = 30, ip: str = "127.0.0.1", port: int = 41451) -> None:
        """
        Connects to AirSim. Wraps the synchronous connection with asyncio.wait_for
        to prevent infinite hanging if AirSim is not running.
        """
        loop = asyncio.get_running_loop()
        try:
            # Use wait_for to enforce the timeout
            self.client = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._connect_sync, ip, port),
                timeout=timeout
            )
            self.connected = True
            self.logger.info("Connected to AirSim.")
        except asyncio.TimeoutError:
            self.last_connect_error = (
                f"AirSim handshake timed out after {timeout} seconds while waiting for confirmConnection()."
            )
            self.logger.warning(self.last_connect_error)
            self.connected = False
        except Exception as exc:
            self.last_connect_error = str(exc)
            self.logger.warning("AirSim connection unavailable. Running in dry mode: %s", exc)
            self.connected = False

    async def print_to_game(self, title: str, message: str) -> None:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor, self._print_to_game_sync, self.client, title, message
            )

    async def reset(self) -> None:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._reset_sync, self.client)
        self.logger.info("Drone environment reset.")

    async def disconnect(self) -> None:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._disconnect_sync, self.client)
        self.logger.info("Drone disconnected.")

    async def takeoff(self) -> None:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._takeoff_sync, self.client)
        self.logger.info("Takeoff command sent.")

    # ── NEW: Velocity ─────────────────────────────────────────────────────────
    def _get_velocity_sync(self, client) -> Dict[str, float]:
        state = client.getMultirotorState()
        vel = state.kinematics_estimated.linear_velocity
        return {"x": vel.x_val, "y": vel.y_val, "z": vel.z_val}

    async def get_velocity(self) -> Dict[str, float]:
        """Returns the drone's current velocity as {x, y, z} in m/s."""
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self._get_velocity_sync, self.client
            )
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    # ── NEW: HUD decision display ──────────────────────────────────────────────
    async def draw_decision(self, text: str, pose: dict = None) -> None:
        """
        Displays the drone's current AI decision on the AirSim HUD (top-left).
        Wraps simPrintLogMessage with a fixed title for clean display.
        `pose` is accepted for API compatibility but unused.
        """
        await self.print_to_game("DRONE AI:", text)

    async def land(self) -> None:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._land_sync, self.client)
        self.logger.info("Land command sent.")

    async def move(self, vx: float, vy: float, vz: float, duration: float) -> None:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor, self._move_sync, self.client, vx, vy, vz, duration
            )
        self.logger.debug("Move command: vx=%.2f vy=%.2f vz=%.2f d=%.2f", vx, vy, vz, duration)

    async def ascend(self, z: float, velocity: float) -> None:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor, self._ascend_sync, self.client, z, velocity
            )
        self.logger.debug("Ascend command: z=%.2f v=%.2f", z, velocity)

    async def hover(self) -> None:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._hover_sync, self.client)
        self.logger.debug("Hover command sent.")

    async def has_collided(self) -> bool:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self._has_collided_sync, self.client
            )
        return False

    async def get_pose(self) -> Dict[str, Any]:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._get_pose_sync, self.client)
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    # ── NEW: Yaw public API ───────────────────────────────────────────────────
    async def get_yaw(self) -> float:
        """
        Returns the current yaw in degrees [-180, 180].
        Returns 0.0 in dry mode (no AirSim connection).
        """
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self._get_yaw_sync, self.client
            )
        return 0.0

    async def rotate_yaw(self, degrees: float, duration: float = 1.5) -> None:
        """
        Rotates the drone by `degrees` relative to its current heading.
          +degrees → turn right (CW)
          -degrees → turn left  (CCW)
        In dry mode the call is silently ignored.
        """
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor, self._rotate_yaw_sync, self.client, degrees, duration
            )
        self.logger.debug("Rotate yaw: %.1f° (duration=%.1fs)", degrees, duration)