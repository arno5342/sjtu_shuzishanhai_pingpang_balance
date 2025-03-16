import pyrealsense2 as rs
import numpy as np
import cv2
import json

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
if __name__ == "__main__":
    check_size=20
    # w h分别是棋盘格模板长边和短边规格（角点个数）
    w = 9
    h = 7
    intrin = {
        "mtx": np.array([[607.961, 0.0, 324.761],
                        [0.0, 607.961, 252.966],
                        [0.0, 0.0, 1.0]],
                        dtype=np.float64),
        "dist": np.array([[0,0,0,0,0]], dtype=np.float64),
    }
    rvec_matrix_sum=np.array([[0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0]])
    sum=0
    pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
    config = rs.config()  # 定义配置config
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)  # 配置depth流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)  # 配置color流
    # 创建对齐对象与color流对齐
    align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
    align = rs.align(align_to)  # rs.align 执行深度帧与其他帧的对齐
    print("start")
    pipe_profile = pipeline.start(config)  # streaming流开始
    while True:
        color_intrin, depth_intrin, img, img_depth, aligned_depth_frame = (
                    get_aligned_images()
                )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # print(cv2.TERM_CRITERIA_EPS,'',cv2.TERM_CRITERIA_MAX_ITER)

        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵，认为在棋盘格这个平面上Z=0
        objp = np.zeros((w * h, 3), np.float32)  # 构造0矩阵，80行3列，用于存放角点的世界坐标
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)  # 三维网格坐标划分

        # 储存棋盘格角点的世界坐标和图像坐标对
        objpoints = []  # 在世界坐标系中的三维点
        imgpoints = []  # 在图像平面的二维点

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 粗略找到棋盘格角点 这里找到的是这张图片中角点的亚像素点位置，共10*8 = 980个点，gray必须是8位灰度或者彩色图，（w,h）为角点规模
        ret, corners = cv2.findChessboardCorners(gray, (w, h))
        # 如果找到足够点对，将其存储起来
        if ret:
            # 精确找到角点坐标
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            worldpoint = objp * check_size  # 棋盘格的宽度
            imagepoint = np.squeeze(corners)  # 将corners降为二维

            (success, rvec, tvec) = cv2.solvePnP(worldpoint, imagepoint, intrin["mtx"], intrin["dist"])
            
            rvec_matrix = cv2.Rodrigues(rvec)[0]
            # 将角点在图像上显示
            cv2.drawChessboardCorners(img, (w, h), corners, ret)
            sum=sum+1
            rvec_matrix_sum=rvec_matrix_sum+rvec_matrix
        cv2.imshow('findCorners', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # 停止流
    pipeline.stop()
    cv2.destroyAllWindows()
    print(rvec_matrix_sum/sum)
    camera_intrin={
        "R":(rvec_matrix_sum/sum).tolist()
    }
    with open("rotation.json","w") as f:
        json.dump(camera_intrin,f)
        print("加载入文件完成...")