#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# import message
from main_yolo_message.msg import DetectionResult
import cv2
from ultralytics import YOLO
import threading
import os

class YOLOServiceServer:
    def __init__(self):
        self.model = YOLO("/home/lajiche/catkin_ws/src/yolov11/pingpang.pt")

        # rospy.loginfo("model ok")
        self.bridge = CvBridge()
        
        self.lock = threading.Lock()  # 用于线程安全
        
        # 订阅图像话题
        # self.usb_cam_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.usb_cam_callback)
        self.realsense_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.realsense_callback)

        # TODO: Topic
        self.pub_detcet = rospy.Publisher("/detection_result", DetectionResult, queue_size=1)

    def process_detection(self, frame):
        """处理检测并返回结果"""
        results = self.model.track(frame, persist=True, verbose=False)
        for r in results:
            if len(r.boxes) > 0:
                for box in r.boxes:
                    # 假设 box 是一个包含时间戳和坐标的数据结构
                    # 例如：timestamp, x, y, z
                    # 这里假设 box.data 是一个列表或元组
                    timestamp = box.data[0]  # 从 box 中读取时间戳
                    x = box.data[1]          # 从 box 中读取 x
                    y = box.data[2]          # 从 box 中读取 y
                    z = box.data[3]          # 从 box 中读取 z
			# 根据矩阵变换得到最终的xyz
                    # 获取类别标签
                    cls_id = int(box.cls[0])
                    label = self.model.names.get(cls_id, "unknown")

                    # 如果检测到球，发布消息
                    # todo: 球的标签？
                    if label == "ball":
                        msg = DetectionResult()
                        msg.header.stamp = rospy.Time.now()
                        msg.header.frame_id = "detected_ball"
                        msg.is_ball_detected = True
                        msg.x = x
                        msg.y = y
                        msg.z = z
                        self.pub_detcet.publish(msg)
                        return
        # 如果没有检测到球，发布消息
        msg = DetectionResult()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "no_ball_detected"
        msg.is_ball_detected = False
        msg.x = 0
        msg.y = 0
        msg.z = 0
        self.pub_detcet.publish(msg)

    def realsense_callback(self, msg):
        # 更新Realsense的最新帧
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        x1, y1, x2, y2 = 400, 0, 900, 620
        with self.lock:
            self.process_detection(cv_image[y1:y2, x1:x2])

if __name__ == "__main__":
    rospy.init_node("yolo_service_server")
    server = YOLOServiceServer()
    rospy.spin()

