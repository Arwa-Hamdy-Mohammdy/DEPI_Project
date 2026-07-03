from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import numpy as np
import cv2
from typing import Any, List, Dict
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor  

# =============================================================================
# Multiprocessing Worker Globals (CPU-heavy YOLO)
# =============================================================================
_process_model = None
_process_weights_path = None
_process_device = None

def _init_vision_worker(weights_path: str, device: str):
    global _process_model, _process_weights_path, _process_device
    _process_weights_path = weights_path
    _process_device = device
    from ultralytics import YOLO
    _process_model = YOLO(weights_path)
    _process_model.predict(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False, device=device)

def _mp_detect_frame(payload: tuple) -> List[Dict[str, Any]]:
    global _process_model, _process_device
    if _process_model is None:
        return []
    frame_bytes, shape, dtype_str, min_conf = payload
    frame = np.frombuffer(frame_bytes, dtype=np.dtype(dtype_str)).copy().reshape(shape)
    results = _process_model(frame, verbose=False, device=_process_device)
    detections: List[Dict[str, Any]] = []
    for result in results:
        names = result.names
        for box in result.boxes:
            conf = float(box.conf.item())
            if conf < min_conf:
                continue
            cls_id = int(box.cls.item())
            coords = box.xyxy[0].tolist()
            detections.append({
                "label": names.get(cls_id, str(cls_id)),
                "confidence": conf,
                "bbox": coords,
            })
    return detections


# =============================================================================
# Async Vision Service
# =============================================================================
class VisionService:
    """Owns a private AirSim camera thread + a process pool for YOLO."""

    # AirSim Segmentation ID Mapping
    SEG_CLASS_MAPPING = {
        10: "person",
        20: "car",
        30: "truck",
        40: "bus",
        50: "building",
        60: "tree",
        70: "fence",
        80: "pole",
        90: "chair",
        100: "pool",
        110: "stairs",
        120: "light"
    }

    def __init__(
        self,
        project_root: Path,
        logger: logging.Logger,
        model_filename: str = "best.pt",
        camera_id: str = "0",
        min_confidence: float = 0.5,
        max_workers: int = 2,
        device: str = "cpu",
        airsim_ip: str = "127.0.0.1",
        airsim_port: int = 41451,
    ):
        self.logger = logger
        self.project_root = project_root
        self.camera_id = camera_id
        self.min_confidence = min_confidence
        self.device = device
        self.airsim_ip = airsim_ip
        self.airsim_port = airsim_port

        # Auto-discover YOLO weights: prefer fine-tuned model, fall back to base
        _model_candidates = [model_filename, "best_updated.pt", "yolo11s.pt", "best.pt", "yolov8n.pt"]
        self.weights_path = None
        for _name in _model_candidates:
            _candidate = project_root / "models" / _name
            if _candidate.exists():
                self.weights_path = _candidate
                break
        if self.weights_path is None:
            # Last resort: check project root
            for _name in ["best_updated.pt", "yolo11s.pt", "best.pt"]:
                _candidate = project_root / _name
                if _candidate.exists():
                    self.weights_path = _candidate
                    break
        if self.weights_path is None:
            self.weights_path = project_root / "models" / model_filename  # will fail gracefully
        self.save_dir = project_root / "data" / "raw_images"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Dedicated thread for AirSim camera RPC (isolated from controller)
        self._airsim_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="airsim_cam")
        self._airsim_client = None

        # Process pool for CPU-heavy YOLO (bypasses Python GIL)
        mp_context = mp.get_context("spawn")
        self._infer_executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=_init_vision_worker,
            initargs=(str(self.weights_path), device),
        )
        self.logger.info(
            "VisionService ready | camera thread: 1 | inference pool: %s workers | device: %s",
            max_workers, device,
        )

    # -------------------------------------------------------------------------
    # AirSim camera helpers (single-threaded)
    # -------------------------------------------------------------------------
    def _init_airsim_sync(self):
        import airsim
        client = airsim.MultirotorClient(ip=self.airsim_ip, port=self.airsim_port)
        client.confirmConnection()
        return client

    async def _get_airsim_client(self):
        if self._airsim_client is None:
            loop = asyncio.get_running_loop()
            self._airsim_client = await loop.run_in_executor(
                self._airsim_executor, self._init_airsim_sync
            )
        return self._airsim_client

    def _fetch_frame_sync(self, client, camera_id: str):
        import airsim
        # Request both RGB and Segmentation
        responses = client.simGetImages([
            airsim.ImageRequest(camera_id, airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(camera_id, airsim.ImageType.Segmentation, False, False)
        ])
        
        results = {"rgb": np.array([]), "seg": np.array([])}
        
        for resp in responses:
            if resp.image_type == airsim.ImageType.Scene:
                results["rgb"] = self._process_response(resp)
            elif resp.image_type == airsim.ImageType.Segmentation:
                results["seg"] = self._process_response(resp)
                
        return results

    def _process_response(self, response):
        if response.width == 0 or response.height == 0:
            return np.array([])
        img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        if img_1d.size == 0:
            return np.array([])
        # AirSim uint8 images are typically 3 channels (BGR)
        return img_1d.reshape(response.height, response.width, 3)

    # -------------------------------------------------------------------------
    # Public async API
    # -------------------------------------------------------------------------
    async def get_frames(self) -> Dict[str, np.ndarray]:
        """Fetch both RGB and Segmentation frames."""
        try:
            client = await self._get_airsim_client()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._airsim_executor, self._fetch_frame_sync, client, self.camera_id
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.error("Failed to retrieve frames: %s", exc)
            return {"rgb": np.array([]), "seg": np.array([])}

    def _extract_seg_objects(self, seg_frame: np.ndarray) -> List[Dict[str, Any]]:
        """Extract bounding boxes from segmentation map for classes YOLO might miss."""
        if seg_frame.size == 0:
            return []
            
        # Convert to grayscale to find unique IDs
        gray = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2GRAY)
        unique_ids = np.unique(gray)
        
        detections = []
        h, w = gray.shape
        
        for obj_id in unique_ids:
            if obj_id == 0 or obj_id not in self.SEG_CLASS_MAPPING:
                continue
                
            label = self.SEG_CLASS_MAPPING[obj_id]
            # Skip objects that YOLO is already good at (to avoid duplicates)
            if any(x in label for x in ["car", "person", "bus", "truck"]):
                continue
                
            # Create mask for this ID
            mask = (gray == obj_id).astype(np.uint8) * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            
            for i in range(1, num_labels):
                x, y, bw, bh, area = stats[i]
                if area < 400: continue # Filter noise
                
                detections.append({
                    "label": label,
                    "confidence": 1.0, # Ground truth from simulator
                    "bbox": [float(x), float(y), float(x+bw), float(y+bh)],
                    "source": "segmentation"
                })
        return detections

    async def get_fused_detections(self, rgb_frame: np.ndarray, seg_frame: np.ndarray) -> List[Dict[str, Any]]:
        """Merge YOLO detections with Segmentation-based detections."""
        # 1. Get YOLO detections (Targets/Dynamic objects)
        yolo_dets = await self.detect_objects(rgb_frame)
        for d in yolo_dets: d["source"] = "yolo"
        
        # 2. Get Segmentation detections (Structures/Environment)
        seg_dets = self._extract_seg_objects(seg_frame)
        
        # 3. Fuse
        return yolo_dets + seg_dets

    async def get_frame(self) -> np.ndarray:
        """Legacy support: returns only RGB frame."""
        frames = await self.get_frames()
        return frames["rgb"]

    async def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Offload YOLO to process pool."""
        if frame.size == 0:
            return []
        try:
            payload = (frame.tobytes(), frame.shape, frame.dtype.str, self.min_confidence)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._infer_executor, _mp_detect_frame, payload)
        except asyncio.CancelledError:
            self.logger.warning("detect_objects cancelled.")
            raise
        except Exception as exc:
            self.logger.error("Detection failed: %s", exc)
            return []

    async def save_detected_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> None:
        """Non-blocking JPEG encode + disk write with Drone Awareness info."""
        if frame.size == 0 or len(detections) == 0:
            return

        def _encode_and_write():
            debug_frame = frame.copy()
            h, w = debug_frame.shape[:2]
            
            # Draw detections
            for obj in detections:
                x1, y1, x2, y2 = map(int, obj["bbox"])
                label = obj['label'].lower()
                conf = obj['confidence']
                
                # Calculate proximity (same logic as StateEncoder)
                size_norm = ((x2 - x1) * (y2 - y1)) / (h * w)
                proximity = min(size_norm * 5.0, 1.0)
                
                # Color based on VisDrone category
                color = (0, 255, 0) # Default Green (Structures/Others)
                if any(x in label for x in ["pedestrian", "people"]):
                    color = (0, 0, 255) # Red for people
                elif any(x in label for x in ["car", "van", "truck", "bus", "vehicle"]):
                    color = (255, 0, 0) # Blue for heavy vehicles
                elif any(x in label for x in ["bicycle", "motor", "tricycle"]):
                    color = (255, 255, 0) # Cyan for light vehicles
                
                # Draw box and label
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} | Prox: {proximity:.2f} | Conf: {conf:.2f}"
                cv2.putText(debug_frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Add "Drone Awareness" overlay header
            cv2.putText(debug_frame, "DRONE SENSING LOG", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = self.save_dir / f"detected_{timestamp}.jpg"
            cv2.imwrite(str(file_path), debug_frame)
            return file_path

        file_path = await asyncio.to_thread(_encode_and_write)
        self.logger.debug("Saved detection log: %s", file_path)

    def shutdown(self):
        self._infer_executor.shutdown(wait=True)
        self._airsim_executor.shutdown(wait=True)
        self.logger.info("VisionService shut down.")

    async def save_frame(self,frame:np.array) -> None:
            if frame.size == 0:
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = self.save_dir / f"detected_{timestamp}.jpg"
            cv2.imwrite(str(file_path), frame)