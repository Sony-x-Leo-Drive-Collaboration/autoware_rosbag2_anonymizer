import os

import torch
import torchvision

import cv2
from PIL import Image

import supervision as sv

import numpy as np

from autoware_rosbag2_anonymizer.model.open_clip import OpenClipModel
from autoware_rosbag2_anonymizer.model.grounding_dino import GroundingDINO
from autoware_rosbag2_anonymizer.model.yolo import Yolo

from autoware_rosbag2_anonymizer.common import (
    create_classes,
    create_yolo_classes,
    bbox_check,
)


class UnifiedLanguageModel:
    def __init__(self, config, json, device) -> None:
        self.device = device

        self.detection_classes, self.classes, self.class_map = create_classes(
            json_data=json
        )

        self.grounding_dino_config_path = config["grounding_dino"]["config_path"]
        self.grounding_dino_checkpoint_path = config["grounding_dino"][
            "checkpoint_path"
        ]

        self.box_threshold = config["grounding_dino"]["box_threshold"]
        self.text_threshold = config["grounding_dino"]["text_threshold"]
        self.nms_threshold = config["grounding_dino"]["nms_threshold"]

        self.open_clip_run = config["openclip"]["run"]
        self.openclip_model_name = config["openclip"]["model_name"]
        self.openclip_pretrained_model = config["openclip"]["pretrained_model"]
        self.openclip_score_threshold = config["openclip"]["score_threshold"]

        self.yolo_model_name = config["yolo"]["model"]
        self.yolo_confidence = config["yolo"]["confidence"]
        self.yolo_config_path = config["yolo"]["config_path"]

        self.iou_threshold = config["bbox_validation"]["iou_threshold"]

        # Grounding DINO
        self.grounding_dino = GroundingDINO(
            self.grounding_dino_config_path,
            self.grounding_dino_checkpoint_path,
            self.device,
        )

        # Openclip
        self.open_clip = OpenClipModel(
            self.openclip_model_name, self.openclip_pretrained_model, self.device
        )

        # YOLO
        self.is_yolo_model_exist = os.path.exists(
            self.yolo_model_name
        ) & os.path.exists(self.yolo_config_path)

        if self.is_yolo_model_exist:
            self.yolo = Yolo(self.yolo_model_name)
            self.yolo_classes = create_yolo_classes(self.yolo_config_path)

    def __call__(self, image: cv2.Mat) -> sv.Detections:
        # Run DINO
        detections = self.grounding_dino(
            image=image,
            classes=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        # Remove class_id if it is 'None'
        not_nons = [
            index
            for index, (_, _, _, class_id, _, _) in enumerate(detections)
            if class_id is not None
        ]
        detections.xyxy = detections.xyxy[not_nons]
        detections.confidence = detections.confidence[not_nons]
        detections.class_id = detections.class_id[not_nons]

        # NMS
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                self.nms_threshold,
            )
            .numpy()
            .tolist()
        )
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # Validation
        valid_ids = []
        invalid_ids = []
        for index, (xyxy, _, score, class_id, _, _) in enumerate(detections):
            if self.classes[class_id] in self.detection_classes:
                is_inside = bbox_check(
                    xyxy,
                    class_id,
                    detections,
                    self.iou_threshold,
                    self.classes,
                    self.class_map,
                )
                
                if not self.open_clip_run and (is_inside or score > 0.25):
                    valid_ids.append(index)
                    continue
                elif self.open_clip_run:
                    # Run OpenClip
                    # and accept as a valid object if the score is greater than 0.9
                    detection_image = image[
                        int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2]), :
                    ]
                    pil_image = Image.fromarray(detection_image)
                    scores = self.open_clip(pil_image, self.classes)
                    if scores.numpy().tolist()[0][class_id] > self.openclip_score_threshold:
                        valid_ids.append(index)
                        continue

                    # Bbox validation
                    # If the object is within the 'should_inside' object
                    # and if the score is the highest among the scores,
                    # or greater than 0.4.
                    is_max = (
                        max(scores.numpy().tolist()[0])
                        == scores.numpy().tolist()[0][class_id]
                    )
                    is_high_03 = scores.numpy().tolist()[0][class_id] > 0.3

                    if is_inside and (is_max or is_high_03):
                        valid_ids.append(index)
                    else:
                        invalid_ids.append(index)
                else:
                    invalid_ids.append(index)
            else:
                invalid_ids.append(index)
        detections.xyxy = detections.xyxy[valid_ids]
        detections.confidence = detections.confidence[valid_ids]
        detections.class_id = np.array(
            [
                self.detection_classes.index(self.classes[x])
                for x in detections.class_id[valid_ids]
            ]
        )

        # Run YOLO
        if self.is_yolo_model_exist:
            yolo_detections = self.yolo(
                image,
                self.yolo_confidence,
            )
            yolo_detections.class_id = np.array(
                [
                    self.detection_classes.index(self.yolo_classes[yolo_id])
                    for yolo_id in yolo_detections.class_id
                ]
            )

            detections = sv.Detections.merge(
                [
                    detections,
                    yolo_detections,
                ]
            )
            detections.class_id = np.array(
                [int(class_id) for _, _, _, class_id, _, _ in detections]
            )

            # # apply NMS again after merge detections
            nms_idx = (
                torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy),
                    torch.from_numpy(detections.confidence),
                    self.nms_threshold,
                )
                .numpy()
                .tolist()
            )
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

        return detections
