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
  tf
  pluginlib
  nav_core
  costmap_2d
  base_local_planner
)

##############################################################################
# 3‧catkin_package（讓其他 package 找得到）
##############################################################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES my_global_planner              # 供 move_base 載入
  CATKIN_DEPENDS roscpp rospy geometry_msgs nav_msgs tf pluginlib nav_core costmap_2d base_local_planner
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
add_library(my_global_planner SHARED src/my_global_planner.cpp)
target_link_libraries(my_global_planner ${catkin_LIBRARIES})

##############################################################################
# 6‧安裝規則
##############################################################################
## 6-1 C++ 可執行檔
#install(TARGETS offboard_control
 # RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

## 6-2 Global-planner 插件(.so)  &  plugin 描述檔 (.xml)
install(TARGETS my_global_planner
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
install(FILES global_planner_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

## 6-3 Python 腳本
catkin_install_python(
  PROGRAMS scripts/limo_patrol.py          # ← 巡邏/避障腳本
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
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
catkin_python_setup()

