import cv2
import numpy as np
from pathlib import Path
import yaml


class AutoLabeler:
    """
    Converts AirSim segmentation masks into YOLO bounding boxes.
    الـ IDs دي اتعملها calibration يدوي بالصور الفعلية.
    """

    CLASS_MAPPING = {
        # ── Skip (background) ──────────────────────────────
        83:  None,   # sky
        103: None,   # road
        120: None,   # ground/grass
        135: None,   # road marking
        153: None,   # غير واضح
        148: None,   # noise

        # 🌳 tree ───────────────────────────────────────────
        84:  "tree",
        119: "tree",
        125: "tree",   # bush
        131: "tree",
        142: "tree",   # vegetation/leaves
        145: "tree",   # hedge
        165: "tree",   # shrub
        171: "tree",

        # 🏠 building ───────────────────────────────────────
        116: "building",
        130: "building",
        138: "building",   # roof
        139: "building",
        147: "building",

        # 🚧 fence/wall ─────────────────────────────────────
        114: "fence",
        132: "fence",
        133: "fence",
        137: "fence",

        # 🚗 car ────────────────────────────────────────────
        128: "car",
        149: "car",

        # 🪧 pole/sign ──────────────────────────────────────
        109: "pole",
        124: "pole",
        129: "pole",
    }

    YOLO_CLASSES = {
        "tree":     0,
        "building": 1,
        "fence":    2,
        "car":      3,
        "pole":     4,
    }

    MIN_AREA = 300

    def __init__(self, dataset_path: str = "data/dataset"):
        self.dataset_path = Path(dataset_path)
        self.seg_dir      = self.dataset_path / "segmentation"
        self.yolo_img_dir = self.dataset_path / "yolo" / "images" / "train"
        self.yolo_lbl_dir = self.dataset_path / "yolo" / "labels" / "train"
        self.yolo_val_img_dir = self.dataset_path / "yolo" / "images" / "val"
        self.yolo_val_lbl_dir = self.dataset_path / "yolo" / "labels" / "val"
        self.yolo_img_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_lbl_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_val_img_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_val_lbl_dir.mkdir(parents=True, exist_ok=True)
        self._stats = {"processed": 0, "skipped": 0, "total_boxes": 0}

    def process_all(self):
        seg_files = list(self.seg_dir.glob("*.png"))
        print(f"📂 لقينا {len(seg_files)} segmentation file")
        for seg_file in seg_files:
            rgb_file = self.dataset_path / "images" / seg_file.name
            if not rgb_file.exists():
                self._stats["skipped"] += 1
                continue
            mask = cv2.imread(str(seg_file))
            if mask is None:
                self._stats["skipped"] += 1
                continue
            bboxes = self._extract_bboxes(mask)
            if bboxes:
                cv2.imwrite(str(self.yolo_img_dir / seg_file.name), cv2.imread(str(rgb_file)))
                lbl_file = self.yolo_lbl_dir / (seg_file.stem + ".txt")
                with open(lbl_file, "w") as f:
                    for b in bboxes:
                        f.write(f"{b['class']} {b['x']:.6f} {b['y']:.6f} {b['w']:.6f} {b['h']:.6f}\n")
                self._stats["processed"] += 1
                self._stats["total_boxes"] += len(bboxes)
            else:
                self._stats["skipped"] += 1
        self._generate_yaml()
        self._print_summary()

    def _extract_bboxes(self, mask: np.ndarray) -> list:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        bboxes = []
        for seg_id, class_name in self.CLASS_MAPPING.items():
            if class_name is None:
                continue
            cls_id = self.YOLO_CLASSES.get(class_name)
            if cls_id is None:
                continue
            binary = (gray == seg_id).astype(np.uint8) * 255
            if binary.sum() == 0:
                continue
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
            for i in range(1, num_labels):
                x, y, bw, bh, area = stats[i]
                if area < self.MIN_AREA:
                    continue
                bboxes.append({
                    "class": cls_id,
                    "x": max(0.0, min(1.0, (x + bw / 2.0) / w)),
                    "y": max(0.0, min(1.0, (y + bh / 2.0) / h)),
                    "w": max(0.0, min(1.0, bw / w)),
                    "h": max(0.0, min(1.0, bh / h)),
                })
        return bboxes

    def _generate_yaml(self):
        yaml_path = self.dataset_path / "yolo" / "data.yaml"
        data = {
            "train": str(self.yolo_img_dir.absolute()),
            "val":   str(self.yolo_val_img_dir.absolute()),
            "nc":    len(self.YOLO_CLASSES),
            "names": list(self.YOLO_CLASSES.keys()),
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ data.yaml → {yaml_path}")

    def _print_summary(self):
        print("\n" + "═" * 45)
        print(f"  ✅ صور اتعالجت   : {self._stats['processed']}")
        print(f"  ⏭️  اتتخطت        : {self._stats['skipped']}")
        print(f"  📦 Boxes المجموع : {self._stats['total_boxes']}")
        print("═" * 45)


if __name__ == "__main__":
    labeler = AutoLabeler(dataset_path="data/dataset")
    labeler.process_all()
    print("✅ Auto-labeling اتخلص!")