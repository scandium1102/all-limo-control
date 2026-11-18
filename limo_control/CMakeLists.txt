cmake_minimum_required(VERSION 3.0.2)
project(limo_control)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  costmap_2d
  diagnostic_updater
  dynamic_reconfigure
  geometry_msgs
  message_filters
  nav_msgs
  nav_core
  pluginlib
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  tf2_ros
  visualization_msgs
)

catkin_python_setup()

generate_dynamic_reconfigure_options(
  cfg/Avoidance.cfg
  cfg/DepthHazard.cfg
)

catkin_package(
  LIBRARIES my_global_planner
  CATKIN_DEPENDS
    costmap_2d
    diagnostic_updater
    dynamic_reconfigure
    geometry_msgs
    message_filters
    nav_msgs
    nav_core
    pluginlib
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tf
    tf2_ros
    visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_library(my_global_planner SHARED src/my_global_planner.cpp)
target_link_libraries(my_global_planner ${catkin_LIBRARIES})

install(TARGETS my_global_planner
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(FILES global_planner_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(
  PROGRAMS
    scripts/depth_avoidance_node.py
    scripts/lidar_avoidance_node.py
    scripts/limo_patrol.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  scripts/record_debug.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

