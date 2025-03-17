#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from main_yolo_service.srv import mainYolo, mainYoloResponse
import cv2
from ultralytics import YOLO
import threading
import os

class YOLOServiceServer:
    def __init__(self):
        # 初始化YOLO模型
        # new_pt_track_path = "/home/lajiche/lajinew_ws/src/yolov11/best_chuansongdai.pt"
        # print("111111111111",os.path.exists(new_pt_track_path))
        # if os.path.exists(new_pt_track_path):
        #     self.model_detect = YOLO(new_pt_track_path)
        # else:
        #     self.model_detect = YOLO("/home/lajiche/lajinew_ws/src/yolov11/rubbish_s_back_track.pt")
        
        # new_pt_classify_path = "/home/lajiche/lajinew_ws/src/yolov11/best_classify.pt"
        # if os.path.exists(new_pt_classify_path):
        #     self.model_classify = YOLO(new_pt_classify_path)
        # else:
        #     self.model_classify = YOLO("/home/lajiche/lajinew_ws/src/yolov11/rubbish_s_just4classify.pt")

        self.model = YOLO("/home/lajiche/catkin_ws/src/yolov11/pingpang.pt")

        # rospy.loginfo("model ok")
        self.bridge = CvBridge()
        
        # 存储最新图像帧
        self.latest_usb_cam_frame = None
        self.latest_realsense_frame = None
        self.lock = threading.Lock()  # 用于线程安全
        
        # 订阅图像话题
        # self.usb_cam_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.usb_cam_callback)
        self.realsense_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.realsense_callback)

        # rospy.loginfo("sub ok")
        # 初始化服务
        self.service = rospy.Service("detection_service", mainYolo, self.handle_service_request)


    def realsense_callback(self, msg):
        # 更新Realsense的最新帧
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        x1, y1, x2, y2 = 400, 0, 900, 620
        with self.lock:
            self.latest_realsense_frame = cv_image[y1:y2, x1:x2]

    def process_detection(self, frame):
        """处理检测并返回结果"""
        results = self.model.track(frame, persist=True, verbose=False)
        labels = []
        t_center = []
        x_center = []
        y_center = []
        z_center = []
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

                    # 填充结果
                    labels.append(label)
                    t_center.append(timestamp)
                    x_center.append(x)
                    y_center.append(y)
                    z_center.append(z)
                return True, labels, t_center, x_center, y_center, z_center
        return False, ["Nodetectedclass"], [], [], [], []


    def handle_service_request(self, req):
        response = mainYoloResponse()
        if req.is_detect_start:
            with self.lock:
                frame = self.latest_realsense_frame
            if frame is not None:
                detected, labels, t, x, y, z = self.process_detection(frame)
                response.is_pingpang_detected = detected
                response.pingpang_class = labels
                response.pingpang_center_t = t
                response.pingpang_center_x = x
                response.pingpang_center_y = y
                response.pingpang_center_z = z
            else:
                response.is_pingpang_detected = False
                response.pingpang_class = ["No frame"]
        else:
            response.is_pingpang_detected = False
            response.pingpang_class = ["Service off"]
        return response

if __name__ == "__main__":
    rospy.init_node("yolo_service_server")
    server = YOLOServiceServer()
    rospy.spin()

