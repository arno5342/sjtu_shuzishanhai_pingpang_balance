cmake_minimum_required(VERSION 3.0.2)
project(main_yolo_service)

# 加载catkin和ROS组件
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

# 添加消息文件声明
add_message_files(
  FILES
  YoloDetection.msg  # 这里添加你的消息文件名，可以有多个
)

# # 服务文件声明
# add_service_files(
#   FILES
#   mainYolo.srv
# )

# 生成消息
generate_messages(
  DEPENDENCIES
  std_msgs
)

# 包配置
catkin_package(
  CATKIN_DEPENDS 
  message_runtime 
  roscpp 
  rospy 
  std_msgs
)

# 指向自定义路径
catkin_install_python(
  PROGRAMS 
  /home/lajiche/catkin_ws/src/main_yolo_service/scripts/detection_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装其他资源文件
install(DIRECTORY
  launch
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
