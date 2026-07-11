from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import queue
import numpy as np
import cv2
from typing import Any, List, Dict
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# Multiprocessing Worker (Persistent Producer-Consumer)
# =============================================================================
def _yolo_worker_loop(
    input_queue: mp.Queue, 
    output_queue: mp.Queue, 
    weights_path: str, 
    device: str, 
    min_confidence: float
):
    """
    Persistent background worker that loads YOLO once and runs continuously.
    It reads frames from the input queue, runs inference, and writes results 
    to the output queue.
    """
    from ultralytics import YOLO
    import traceback
    
    try:
        model = YOLO(weights_path)
        # Warmup
        model.predict(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False, device=device)
    except Exception as e:
        print(f"YOLO Worker failed to initialize: {e}")
        return

    while True:
        try:
            # Block until a frame is available
            payload = input_queue.get()
            if payload is None:  # Poison pill for graceful shutdown
                break
                
            frame_bytes, shape, dtype_str = payload
            frame = np.frombuffer(frame_bytes, dtype=np.dtype(dtype_str)).copy().reshape(shape)
            
            # Inference
            results = model(frame, verbose=False, device=device, augment=True, imgsz=640)
            
            detections: List[Dict[str, Any]] = []
            for result in results:
                names = result.names
                for box in result.boxes:
                    conf = float(box.conf.item())
                    if conf < min_confidence:
                        continue
                    cls_id = int(box.cls.item())
                    coords = box.xyxy[0].tolist()
                    detections.append({
                        "label": names.get(cls_id, str(cls_id)),
                        "confidence": conf,
                        "bbox": coords,
                    })
                    
            # Push to output queue. 
            # We drain the output queue first to prevent memory bloat and ensure only the absolute freshest frame is read by the main loop.
            while not output_queue.empty():
                try: output_queue.get_nowait()
                except queue.Empty: break
                
            output_queue.put(detections)
            
        except Exception as e:
            print(f"YOLO Worker encountered an error during inference: {traceback.format_exc()}")
            continue


# =============================================================================
# Async Vision Service
# =============================================================================
class VisionService:
    """Owns a private AirSim camera thread + a persistent YOLO process."""

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
        max_workers: int = 1, # Unused now, keeping for interface compatibility
        device: str = "cuda",
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

        # Persistent Process Setup (Producer-Consumer pattern)
        mp_context = mp.get_context("spawn")
        self.input_queue = mp_context.Queue(maxsize=2)
        self.output_queue = mp_context.Queue(maxsize=2)
        self.latest_yolo_detections: List[Dict[str, Any]] = []
        
        self.worker_process = mp_context.Process(
            target=_yolo_worker_loop,
            args=(
                self.input_queue, 
                self.output_queue, 
                str(self.weights_path), 
                self.device, 
                self.min_confidence
            ),
            daemon=True # Ensures the process exits when the main script crashes
        )
        self.worker_process.start()
        
        self.logger.info(
            "VisionService ready | camera thread: 1 | persistent YOLO process: 1 | device: %s",
            device,
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
        # Request RGB, Segmentation, and DepthPlanar
        responses = client.simGetImages([
            airsim.ImageRequest(camera_id, airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(camera_id, airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest(camera_id, airsim.ImageType.DepthPlanar, True, False)
        ])
        
        results = {"rgb": np.array([]), "seg": np.array([]), "depth": np.array([])}
        
        for resp in responses:
            if resp.image_type == airsim.ImageType.Scene:
                results["rgb"] = self._process_response(resp)
            elif resp.image_type == airsim.ImageType.Segmentation:
                results["seg"] = self._process_response(resp)
            elif resp.image_type == airsim.ImageType.DepthPlanar:
                if resp.width > 0 and resp.height > 0:
                    depth_raw = np.array(resp.image_data_float, dtype=np.float32).reshape(resp.height, resp.width)
                    results["depth"] = depth_raw
                
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
        """Merge YOLO detections with Segmentation-based detections in a non-blocking way."""
        # 1. Producer: Send latest RGB frame to the YOLO worker if it is ready
        if rgb_frame.size > 0:
            try:
                payload = (rgb_frame.tobytes(), rgb_frame.shape, rgb_frame.dtype.str)
                self.input_queue.put_nowait(payload)
            except queue.Full:
                pass # YOLO is still processing, drop frame (don't block the control loop)

        # 2. Consumer: Drain output queue to fetch the absolute freshest YOLO detection
        new_dets = None
        while not self.output_queue.empty():
            try:
                new_dets = self.output_queue.get_nowait()
            except queue.Empty:
                break
        
        if new_dets is not None:
            self.latest_yolo_detections = new_dets

        # 3. Process fast Segmentation logic (runs on main thread, ~1-3ms)
        seg_dets = self._extract_seg_objects(seg_frame)
        
        # 4. Fuse the latest known YOLO state with the real-time Segmentation state
        yolo_dets_copy = list(self.latest_yolo_detections)
        for d in yolo_dets_copy: 
            d["source"] = "yolo"
            
        return yolo_dets_copy + seg_dets

    async def get_frame(self) -> np.ndarray:
        """Legacy support: returns only RGB frame."""
        frames = await self.get_frames()
        return frames["rgb"]

    async def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Legacy compatibility method. Defers to the persistent worker and returns latest."""
        if frame.size > 0:
            try:
                payload = (frame.tobytes(), frame.shape, frame.dtype.str)
                self.input_queue.put_nowait(payload)
            except queue.Full:
                pass
                
        # Immediately return the cached result without blocking
        return list(self.latest_yolo_detections)

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
        """Gracefully shut down the persistent worker and thread pool."""
        self.logger.info("Initiating VisionService shutdown...")
        try:
            self.input_queue.put_nowait(None) # Poison pill
        except queue.Full:
            pass
            
        self.worker_process.join(timeout=2.0)
        if self.worker_process.is_alive():
            self.worker_process.terminate()
            
        self._airsim_executor.shutdown(wait=True)
        self.logger.info("VisionService shut down successfully.")

    async def save_frame(self,frame:np.array) -> None:
            if frame.size == 0:
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = self.save_dir / f"detected_{timestamp}.jpg"
            cv2.imwrite(str(file_path), frame)