"""
data_collector.py
=================
يجمع صور RGB + Segmentation من AirSim لتدريب YOLO.
شغّله وهو متصل بـ AirSim → هيحفظ في data/dataset/
"""

import airsim
import time
import cv2
import numpy as np
import random
from pathlib import Path


class DataCollector:

    def __init__(self, output_dir: str = "data/dataset"):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.output_dir = Path(output_dir)
        self.img_dir    = self.output_dir / "images"
        self.seg_dir    = self.output_dir / "segmentation"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.seg_dir.mkdir(parents=True, exist_ok=True)

    def collect_step(self, frame_idx: int) -> bool:
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene,        False, False),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
        ])
        if len(responses) < 2:
            return False

        img_rgb = self._to_bgr(responses[0])
        img_seg = self._to_bgr(responses[1])
        if img_rgb is None or img_seg is None:
            return False

        ts   = int(time.time() * 1000)
        name = f"frame_{frame_idx:05d}_{ts}.png"
        cv2.imwrite(str(self.img_dir / name), img_rgb)
        cv2.imwrite(str(self.seg_dir / name), img_seg)
        return True

    def _to_bgr(self, response) -> np.ndarray | None:
        if response.width == 0 or response.height == 0:
            return None
        arr = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        if arr.size == 0:
            return None

        expected_rgba = response.height * response.width * 4
        expected_rgb  = response.height * response.width * 3

        if arr.size == expected_rgba:
            img = arr.reshape(response.height, response.width, 4)
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif arr.size == expected_rgb:
            img = arr.reshape(response.height, response.width, 3)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            print(f"⚠️ Unexpected buffer size: {arr.size}")
            return None

    def run_exploration(self, num_frames: int = 500):
        print("🛫 Takeoff...")
        self.client.takeoffAsync().join()

        # ارتفاعات مختلفة عشان نشوف trees وcars
        altitudes = [-3, -5, -8, -12, -15]

        captured = 0
        for i in range(num_frames):
            # كل 50 فريم: غيّر الارتفاع
            if i % 50 == 0:
                alt = altitudes[(i // 50) % len(altitudes)]
                self.client.moveToZAsync(alt, 3).join()
                print(f"  🔁 Altitude changed to {alt}m")

            vx       = random.uniform(-4, 4)
            vy       = random.uniform(-4, 4)
            yaw_rate = random.uniform(-25, 25)

            self.client.moveByVelocityAsync(
                vx, vy, 0.0, 0.6,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(True, yaw_rate)
            ).join()

            if self.collect_step(i):
                captured += 1
                if captured % 20 == 0:
                    print(f"  📸 Captured {captured}/{num_frames} frames")

        print(f"\n✅ Done! Captured {captured} frames")
        self.client.landAsync().join()


if __name__ == "__main__":
    collector = DataCollector(output_dir="data/dataset")
    collector.run_exploration(num_frames=500)