import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import datetime
import json
print("start0")
def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

    #### 获取相机参数 ####
    depth_intrin = (
        aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    )  # 获取深度参数（像素坐标系转相机坐标系会用到）

    color_intrin = (
        aligned_color_frame.profile.as_video_stream_profile().intrinsics
    )  # 获取相机内参

    #### 将images转为numpy arrays ####
    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame
def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis+BALL_RADIUS)
    return dis, camera_coordinate

if __name__ == "__main__":
    WORLD2CAMERA=False
    with open("rotation.json",'r') as file:
        dict=json.load(file)
    R=np.array(dict["R"])#读取相机外参
    filename = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S").replace(":", "_") + ".txt"
    )#创建文件
    ov_model = YOLO(".\pingpang_openvino_model")
    BALL_RADIUS = 0.02  # 球半径，单位m
    z_=0.04
    x_=0.48
    y_=0.4# 机器臂参数
    pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
    config = rs.config()  # 定义配置config
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)  # 配置depth流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)  # 配置color流
    # 创建对齐对象与color流对齐
    align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
    align = rs.align(align_to)  # rs.align 执行深度帧与其他帧的对齐
    print("start")
    with open(filename, "a") as file:
        pipe_profile = pipeline.start(config)  # streaming流开始
        while True:
            color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = (
                        get_aligned_images()
                    )
            results = ov_model(img_color)
            if len(results[0].boxes)>0:
                result = results[0].boxes[0] # 访问检测结果
                x1, y1, x2, y2 = result.xyxy[0]
                confidence = result.conf[0]
                cls = result.cls[0]
                label = ov_model.names[int(cls)]

                # 计算中心点

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                 # 获取中心点的三维坐标

                dis, camera_coordinate = get_3d_camera_coordinate(
                    [center_x, center_y], aligned_depth_frame, depth_intrin
                )
                camera_coordinate=np.dot(camera_coordinate,R)#camera2world onlt rotation
                while WORLD2CAMERA==False:
                    orignepose=camera_coordinate
                    WORLD2CAMERA=True
                print(camera_coordinate,orignepose)
                camera_coordinate=[camera_coordinate[i]-orignepose[i] for i in range(len(camera_coordinate))]
                print(camera_coordinate)
                camera_coordinate=[camera_coordinate[i]+[x_,y_,z_][i] for i in range(len(camera_coordinate))]
                #camera2world only transition
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                file.write(
                    f"{timestamp}, X: {camera_coordinate[0]:.2f}, Y: {camera_coordinate[1]:.2f}, Z: {camera_coordinate[2]:.2f}\n"
                )#写入文件
            cv2.imshow("YOLO Detection", img_color)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            # 停止流
        pipeline.stop()
        cv2.destroyAllWindows()