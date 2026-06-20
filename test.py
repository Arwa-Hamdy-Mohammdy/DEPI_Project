from mask import mask_v4
from pathlib import Path
import logging
from typing import List
import numpy as np
import cv2

def __get_imgs(path:str) -> List[Path]:
    p = Path(path)

    imgs_extensions = [".png",".jpg",".jpeg"]

    imgs = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in imgs_extensions]
    
    if len(imgs) == 0:
        logging.warning("images not found in directory")
        return []
    
    return imgs

if __name__ == "__main__":
    OUTPUT_DIR = Path(r"D:\Computer_Science\New folder (2)\output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    imgs = __get_imgs(r"D:\Computer_Science\New folder (2)")

    m = mask_v4.DroneVisionPipeline(model_path="best_updated.pt",confidence=0.5)

    for counter,img_path in enumerate(imgs,start=1):

        frame = cv2.imread(str(img_path))

        if frame is None:
            logging.warning(f"Failed to read image: {img_path.name}")
            continue

        result, mask, masked_frame = m.process_frame(frame)

        logging.info(f"details: {result}")

        frame_path = OUTPUT_DIR / f"drone_{counter}.jpg"
        masked_frame_path = OUTPUT_DIR / f"drone_masked_{counter}.jpg"

        success1 = cv2.imwrite(str(frame_path),frame)
        success2 = cv2.imwrite(str(masked_frame_path),masked_frame)

        if success1 and success2:
            logging.info(f"[INFO] | Photo save successfully in {frame_path}")
            logging.info(f"[INFO] | Photo save successfully in {masked_frame_path}")
        else:
            logging.warning(f"[WARNING] | Failed to save photo: {OUTPUT_DIR}")
