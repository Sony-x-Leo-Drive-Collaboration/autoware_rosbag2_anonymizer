import rosbag2_py
from rclpy.serialization import serialize_message

from cv_bridge import CvBridge
import cv2

from autoware_rosbag2_anonymizer.common import ENCODINGS
from autoware_rosbag2_anonymizer.rosbag_io.rosbag_common import (
    get_rosbag_options,
    create_topic,
)


class RosbagWriter:
    def __init__(
        self,
        bag_path: str,
        write_compressed: bool,
        storage_id: str,
        offered_qos_profiles_map: dict,
    ) -> None:
        self.bag_path = bag_path
        self.write_compressed = write_compressed
        self.offered_qos_profiles_map = offered_qos_profiles_map

        self.storage_id = storage_id

        storage_options, converter_options = get_rosbag_options(
            self.bag_path, self.storage_id
        )
        self.writer = rosbag2_py.SequentialWriter()
        self.writer.open(storage_options, converter_options)

        self.type_map = {}

        self.bride = CvBridge()

    def __dell__(self):
        self.writer.close()

    def write_image(self, image, topic_name, timestamp, encoding="bgr8"):
        if topic_name not in self.type_map:
            create_topic(
                self.writer,
                topic_name,
                (
                    "sensor_msgs/msg/Image"
                    if not self.write_compressed
                    else "sensor_msgs/msg/CompressedImage"
                ),
                "cdr",
                self.offered_qos_profiles_map[topic_name],
            )
            self.type_map[topic_name] = (
                "sensor_msgs/msg/Image"
                if not self.write_compressed
                else "sensor_msgs/msg/CompressedImage"
            )

        if self.write_compressed or encoding == "bgr8":
            image_msg = self.bride.cv2_to_compressed_imgmsg(image)
        else:
            image_msg = self.bride.cv2_to_imgmsg(
                cv2.cvtColor(image, ENCODINGS[encoding]["backward"])
            )
            image_msg._encoding = encoding 

        image_msg.header.stamp.sec = timestamp // 10**9 
        image_msg.header.stamp.nanosec = timestamp % 10**9 

        self.writer.write(topic_name, serialize_message(image_msg), timestamp)
        
    def write_any(self, msg, msg_type, topic_name, timestamp):
        if topic_name not in self.type_map:
            create_topic(
                self.writer,
                topic_name,
                msg_type,
                "cdr",
                self.offered_qos_profiles_map[topic_name],
            )

        self.writer.write(topic_name, serialize_message(msg), timestamp)
