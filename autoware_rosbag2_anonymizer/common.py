from typing import List, Tuple, Dict

import cv2

import os
import sys
import yaml

import supervision as sv

ENCODINGS = {
    "rgb8": {
        "forward": cv2.COLOR_RGB2BGR,
        "backward": cv2.COLOR_BGR2RGB,
    },
    "rgba8": {
        "forward": cv2.COLOR_RGBA2BGR,
        "backward": cv2.COLOR_BGR2RGBA,
    },
}


def get_file_paths(root_folder: str, extensions: List[str]):
    file_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    file_paths.append(os.path.join(root, file))
                    break
    return file_paths


def create_classes(json_data) -> Tuple[List[str], List[str], Dict]:
    detection_classes = []
    prompts = []
    prompts_map = {}

    for data in json_data["prompts"]:
        if data["prompt"] not in detection_classes:
            detection_classes.append(data["prompt"])

    for data in json_data["prompts"]:
        if data["prompt"] not in prompts:
            prompts.append(data["prompt"])
        [
            prompts.append(prompt)
            for prompt in data["should_inside"]
            if prompt not in prompts
        ]
        [
            prompts.append(prompt)
            for prompt in data["should_not_inside"]
            if prompt not in prompts
        ]

    for data in json_data["prompts"]:
        prompts_map[data["prompt"]] = {
            "should_inside": data["should_inside"],
            "should_not_inside": data["should_not_inside"],
        }

    return detection_classes, prompts, prompts_map


def create_yolo_classes(yolo_config_path) -> List[str]:
    with open(yolo_config_path, "r") as file:
        yolo_config = yaml.safe_load(file)

    return yolo_config["names"]


def calculate_iou(box1, box2) -> int:
    A1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    A2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    if x2_i > x1_i and y2_i > y1_i:
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        iou = inter_area / min(A1, A2)
        return iou
    else:
        return 0.0


def bbox_check(xyxy, class_id, detections, iou_threshold, classes, class_map) -> bool:
    for valid_xyxy, _, _, valid_class_id, _, _ in detections:
        if classes[valid_class_id] in class_map[classes[class_id]]["should_inside"]:
            iou = calculate_iou(xyxy, valid_xyxy)
            if iou >= iou_threshold:
                return True
    return False


def blur_detections(img, detections, kernel_size, sigma_x):
    output_img = img.copy()
    for xyxy, mask, _, _, _, _ in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        region_to_blur = img[y1:y2, x1:x2]
        mask_cropped = mask[y1:y2, x1:x2]

        blurred_region = cv2.GaussianBlur(
            region_to_blur, (kernel_size, kernel_size), sigma_x
        )
        output_img[y1:y2, x1:x2][mask_cropped] = blurred_region[mask_cropped]
    return output_img


def split_dataset() -> None:
    if len(sys.argv) != 2:
        print("Usage: autoware-rosbag2-anonymizer-split-dataset <dataset-path>")
        return

    dataset_path = sys.argv[1]

    ds = sv.DetectionDataset.from_yolo(
        images_directory_path=os.path.join(dataset_path, "images"),
        annotations_directory_path=os.path.join(dataset_path, "annotations"),
        data_yaml_path=os.path.join(dataset_path, "data.yaml"),
    )

    train_ds, val_ds = ds.split(split_ratio=0.7, random_state=42, shuffle=True)

    train_ds.as_yolo(
        os.path.join(dataset_path, "train/images"),
        os.path.join(dataset_path, "train/labels"),
    )
    val_ds.as_yolo(
        os.path.join(dataset_path, "valid/images"),
        os.path.join(dataset_path, "valid/labels"),
    )

    with open(os.path.join(dataset_path, "data.yaml"), "r") as f:
        yaml_content = yaml.safe_load(f)

    data = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
    }
    yaml_content.update(data)

    with open(os.path.join(dataset_path, "data.yaml"), "w") as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)
