"""Prepare TP-Dataset into YOLO detection and segmentation layouts."""

import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from fix_dataset import flatten_dataset


CLASS_INDEX = 0
WORKSPACE_ROOT = Path(__file__).parent
DATASET_ROOT = WORKSPACE_ROOT / "data" / "TP-Dataset"
GROUND_TRUTH_DIR = DATASET_ROOT / "GroundTruth"
JPEG_IMAGES_DIR = DATASET_ROOT / "JPEGImages"
INDEX_DIR = DATASET_ROOT / "Index"
YOLO_CONTOUR_LABEL_DIR = DATASET_ROOT / "YOLO_ContourLabel"
YOLO_BOX_LABEL_DIR = DATASET_ROOT / "YOLO_BoxLabel"
YOLO_DATA_DIR = DATASET_ROOT / "YOLO_Data"


def unwrap(obj, msg: str = "Unexpected None"):
    if obj is None:
        raise ValueError(msg)
    return obj


def path_traversal(base_path: Path):
    for path in sorted(base_path.rglob("*.png")):
        yield path, path.relative_to(base_path)


def simplify_contour(contour, epsilon_ratio: float = 0.002):
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def write_yolo_labels():
    YOLO_CONTOUR_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_BOX_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    for file_path, file_name in tqdm(path_traversal(GROUND_TRUTH_DIR), desc="Generating TP labels"):
        image = unwrap(cv2.imread(str(file_path)), f"Failed to read {file_path}")
        binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [simplify_contour(contour) for contour in contours if len(contour) > 0]
        if not contours:
            continue

        contour_points = np.vstack(contours)
        contour_points_x = contour_points[:, 0, 0]
        contour_points_y = contour_points[:, 0, 1]
        x_min = int(contour_points_x.min())
        y_min = int(contour_points_y.min())
        x_max = int(contour_points_x.max())
        y_max = int(contour_points_y.max())

        height, width = image.shape[:2]
        normalized_contour_points = [
            [point[0][0] / width, point[0][1] / height] for point in contour_points
        ]
        normalized_box = (
            x_min / width,
            y_min / height,
            x_max / width,
            y_max / height,
        )

        relative_name = file_name.as_posix()
        contour_label_path = YOLO_CONTOUR_LABEL_DIR / Path(relative_name).with_suffix(".txt")
        contour_label_path.parent.mkdir(parents=True, exist_ok=True)
        contour_point_str = " ".join(
            f"{point[0]:.6f} {point[1]:.6f}" for point in normalized_contour_points
        )
        contour_label_path.write_text(f"{CLASS_INDEX} {contour_point_str}\n")

        x_min_norm, y_min_norm, x_max_norm, y_max_norm = normalized_box
        x_center = (x_min_norm + x_max_norm) / 2
        y_center = (y_min_norm + y_max_norm) / 2
        box_width = x_max_norm - x_min_norm
        box_height = y_max_norm - y_min_norm

        box_label_path = YOLO_BOX_LABEL_DIR / Path(relative_name).with_suffix(".txt")
        box_label_path.parent.mkdir(parents=True, exist_ok=True)
        box_label_path.write_text(
            f"{CLASS_INDEX} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
        )


def read_split_file(path: Path):
    return path.read_text().splitlines()


def copy_data(file_list, image_src: Path, label_src: Path, yolo_dir: Path):
    for file_base_name in tqdm(file_list, desc=f"Copying to {yolo_dir.name}"):
        image_path = image_src / f"{file_base_name}.jpg"
        label_path = label_src / f"{file_base_name}.txt"

        dst_image_path = yolo_dir / "images" / f"{file_base_name}.jpg"
        dst_label_path = yolo_dir / "labels" / f"{file_base_name}.txt"

        dst_image_path.parent.mkdir(parents=True, exist_ok=True)
        dst_label_path.parent.mkdir(parents=True, exist_ok=True)

        if not image_path.exists() or not label_path.exists():
            print(f"Skipping missing pair: {file_base_name}")
            continue

        shutil.copy2(image_path, dst_image_path)
        shutil.copy2(label_path, dst_label_path)


def prepare_yolo_data(yolo_base_dir: Path, train_data, val_data, test_data, image_src: Path, label_src: Path):
    copy_data(train_data, image_src, label_src, yolo_base_dir / "train")
    copy_data(val_data, image_src, label_src, yolo_base_dir / "val")
    copy_data(test_data, image_src, label_src, yolo_base_dir / "test")


def main():
    write_yolo_labels()

    train_data = read_split_file(INDEX_DIR / "train.txt")
    val_data = read_split_file(INDEX_DIR / "val.txt")
    test_data = read_split_file(INDEX_DIR / "predict.txt")

    yolo_box_data = YOLO_DATA_DIR / "boxes"
    yolo_contour_data = YOLO_DATA_DIR / "contours"
    yolo_box_data.mkdir(parents=True, exist_ok=True)
    yolo_contour_data.mkdir(parents=True, exist_ok=True)

    prepare_yolo_data(yolo_box_data, train_data, val_data, test_data, JPEG_IMAGES_DIR, YOLO_BOX_LABEL_DIR)
    prepare_yolo_data(yolo_contour_data, train_data, val_data, test_data, JPEG_IMAGES_DIR, YOLO_CONTOUR_LABEL_DIR)

    flatten_dataset(yolo_box_data, YOLO_DATA_DIR / "boxes_fixed", fix_labels=False)
    flatten_dataset(yolo_contour_data, YOLO_DATA_DIR / "contours_fixed", fix_labels=False)

    print("TP-Dataset YOLO preparation complete.")
    print(f"Detection dataset: {YOLO_DATA_DIR / 'boxes_fixed'}")
    print(f"Segmentation dataset: {YOLO_DATA_DIR / 'contours_fixed'}")


if __name__ == "__main__":
    main()