##############################################################################
# 1‧基本資訊
##############################################################################
cmake_minimum_required(VERSION 3.0.2)
project(limo_control)

# 建議用 C++14 以上
add_compile_options(-std=c++14)

##############################################################################
# 2‧相依套件
##############################################################################
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  diagnostic_updater
  dynamic_reconfigure
  tf2_ros
  tf2_geometry_msgs
  tf
  pluginlib
  nav_core
  costmap_2d
  base_local_planner
  message_filters
  image_transport
  camera_info_manager
  depth_image_proc
  pcl_ros
  pcl_conversions
  visualization_msgs
  actionlib
  actionlib_msgs
  move_base_msgs
  message_generation
)

set(_limo_drcfg_files
  ${CMAKE_CURRENT_SOURCE_DIR}/cfg/Avoidance.cfg
  ${CMAKE_CURRENT_SOURCE_DIR}/cfg/DepthHazard.cfg
)

add_service_files(
  FILES
    GoToRoom.srv
)

file(MAKE_DIRECTORY
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}/cfg
)

set(limo_control_CFG_PY "")
set(limo_control_CFG_HDR "")
foreach(_cfg ${_limo_drcfg_files})
  get_filename_component(_cfg_name ${_cfg} NAME_WE)
  set(_cfg_py ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}/cfg/${_cfg_name}Config.py)
  set(_cfg_hdr ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_INCLUDE_DESTINATION}/${_cfg_name}Config.h)
  add_custom_command(
    OUTPUT ${_cfg_py} ${_cfg_hdr}
    COMMAND ${PYTHON_EXECUTABLE} ${_cfg} ${dynamic_reconfigure_PACKAGE_PATH}
            ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}
            ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_INCLUDE_DESTINATION}
            ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
    DEPENDS ${_cfg}
    COMMENT "Generating dynamic reconfigure files for ${_cfg_name}"
  )
  list(APPEND limo_control_CFG_PY ${_cfg_py})
  list(APPEND limo_control_CFG_HDR ${_cfg_hdr})
endforeach()

add_custom_target(limo_control_gencfg ALL DEPENDS ${limo_control_CFG_PY} ${limo_control_CFG_HDR})

generate_messages(
  DEPENDENCIES
    std_msgs
    geometry_msgs
    actionlib_msgs
)

##############################################################################
# 3‧catkin_package（讓其他 package 找得到）
##############################################################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES my_global_planner              # 供 move_base 載入
  EXPORTED_TARGETS limo_control_gencfg ${${PROJECT_NAME}_EXPORTED_TARGETS}
  CATKIN_DEPENDS
    roscpp
    rospy
    geometry_msgs
    nav_msgs
    sensor_msgs
    std_msgs
    diagnostic_updater
    dynamic_reconfigure
    tf
    tf2_ros
    tf2_geometry_msgs
    pluginlib
    nav_core
    costmap_2d
    base_local_planner
    message_filters
    image_transport
    camera_info_manager
    depth_image_proc
    pcl_ros
    pcl_conversions
    visualization_msgs
    actionlib
    actionlib_msgs
    move_base_msgs
    message_runtime
    ros_numpy
)

##############################################################################
# 4‧Include 目錄
##############################################################################
include_directories(
  include                                   # 若有 header 檔
  ${catkin_INCLUDE_DIRS}
)

##############################################################################
# 5‧可執行檔 / 函式庫
##############################################################################
## (1) 範例 C++ Off-board 控制節點（如無此檔可刪）
#add_executable(offboard_control src/offboard_control.cpp)
#target_link_libraries(offboard_control ${catkin_LIBRARIES})

## (2) ★ A* 全域規劃器插件  ─ 必須是 SHARED library ★
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/my_global_planner.cpp")
  add_library(my_global_planner SHARED src/my_global_planner.cpp)
  target_link_libraries(my_global_planner ${catkin_LIBRARIES})
  set(HAS_MY_GLOBAL_PLANNER TRUE)
else()
  message(STATUS "Skipping my_global_planner: src/my_global_planner.cpp not found")
  set(HAS_MY_GLOBAL_PLANNER FALSE)
endif()

##############################################################################
# 6‧安裝規則
##############################################################################
## 6-1 C++ 可執行檔
#install(TARGETS offboard_control
 # RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

## 6-2 Global-planner 插件(.so)  &  plugin 描述檔 (.xml)
if(HAS_MY_GLOBAL_PLANNER)
  install(TARGETS my_global_planner
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
  install(FILES global_planner_plugin.xml
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
endif()

install(FILES ${limo_control_CFG_HDR}
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

install(DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}/cfg
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION})

## 6-3 Python 腳本
catkin_install_python(
  PROGRAMS
    scripts/limo_patrol.py          # ← 巡邏/避障腳本
    scripts/lidar_avoidance_node.py
    scripts/depth_avoidance_node.py
    scripts/cmd_vel_arbiter.py
    scripts/cmd_vel_safety_filter.py
    scripts/explore_supervisor.py
    scripts/room_segmenter.py
    scripts/room_navigator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  scripts/record_debug.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY scripts/patrol_modules/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/patrol_modules
  FILES_MATCHING PATTERN "*.py"
)

## 6-4 Launch、Config 等（如有其他檔可自行追加）
install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

##############################################################################
# 7‧單元測試 (可省略)
##############################################################################
# if(CATKIN_ENABLE_TESTING)
#   find_package(rostest REQUIRED)
#   add_rostest(test/xxx.test)
# endif()
# catkin_python_setup()  # no setup.py; avoid redundant dependency
