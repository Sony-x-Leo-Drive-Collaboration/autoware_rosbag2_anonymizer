import cv2
import cv_bridge

import supervision as sv

from autoware_rosbag2_anonymizer.common import (
    create_yolo_classes,
    blur_detections,
    ENCODINGS,
)

from autoware_rosbag2_anonymizer.model.yolo import Yolo
from autoware_rosbag2_anonymizer.model.sam2 import SAM2

from autoware_rosbag2_anonymizer.rosbag_io.rosbag_reader import RosbagReader
from autoware_rosbag2_anonymizer.rosbag_io.rosbag_writer import RosbagWriter


def yolo_anonymize(config_data, json_data, device) -> None:

    yolo_model_path = config_data["yolo"]["model"]
    yolo_confidence = config_data["yolo"]["confidence"]
    yolo_config_path = config_data["yolo"]["config_path"]

    # YOLO
    yolo_model = Yolo(yolo_model_path)

    # Declare classes for YOLO from yaml file
    CLASSES = create_yolo_classes(yolo_config_path)

    # Segment-Anything-2
    SAM2_MODEL_CFG = config_data["segment_anything_2"]["model_cfg"]
    SAM_CHECKPOINT_PATH = config_data["segment_anything_2"]["checkpoint_path"]
    sam2 = SAM2(SAM2_MODEL_CFG, SAM_CHECKPOINT_PATH, device)

    # Create rosbag reader and rosbag writer
    reader = RosbagReader(config_data["rosbag"]["input_bag_path"], 1)
    writer = RosbagWriter(
        config_data["rosbag"]["output_bag_path"],
        config_data["rosbag"]["output_save_compressed_image"],
        config_data["rosbag"]["output_storage_id"],
        reader.get_qos_profile_map(),
    )

    for i, (msg, is_image) in enumerate(reader):
        if not is_image:
            writer.write_any(msg.data, msg.type, msg.topic, msg.timestamp)
        else:
            if "Compressed" in msg.type:
                image = cv_bridge.CvBridge().compressed_imgmsg_to_cv2(msg.data)
            else:
                image = cv_bridge.CvBridge().imgmsg_to_cv2(msg.data)
                if msg.data._encoding != "bgr8":
                    image = cv2.cvtColor(
                        image, ENCODINGS[msg.data._encoding]["forward"]
                    )

            detections = yolo_model(image, confidence=yolo_confidence)

            # Run SAM
            if config_data["blur"]["region"] == "mask":
                detections = sam2(image=image, detections=detections)

            # Blur detections
            output = blur_detections(
                image,
                detections,
                config_data["blur"]["region"],
                config_data["blur"]["kernel_size"],
                config_data["blur"]["sigma_x"],
            )

            # Write blured image to rosbag
            if "Compressed" in msg.type:
                writer.write_image(output, msg.topic, msg.timestamp)
            else:
                writer.write_image(output, msg.topic, msg.timestamp, msg.data._encoding)

            # Print detections: how many objects are detected in each class
            if config_data["debug"]["print_on_terminal"]:
                print("\nDetections:")
                for class_id in range(len(CLASSES)):
                    print(
                        f"{CLASSES[class_id]}: {len([d for d in detections if d[3] == class_id])}"
                    )

            # Show debug image
            if config_data["debug"]["show_on_image"]:
                bounding_box_annotator = sv.BoxAnnotator()
                annotated_image = bounding_box_annotator.annotate(
                    scene=output,
                    detections=detections,
                )

                labels = [
                    f"{CLASSES[class_id]} {confidence:0.2f}"
                    for _, _, confidence, class_id, _, _ in detections
                ]
                label_annotator = sv.LabelAnnotator()
                annotated_image = label_annotator.annotate(
                    output,
                    detections,
                    labels,
                )

                height, width = image.shape[:2]
                annotated_image = cv2.resize(annotated_image, (width // 2, height // 2))
                cv2.imshow("test", annotated_image)
                cv2.waitKey(1)
