#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
limo_patrol.py  ─ LIMO 巡邏＋避障腳本 (LiDAR + Depth fusion)
作者：ChatGPT assist  2025-07-xx
"""

import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math
import time

# ────────────────────────────────────────────────────────
# 可調參數  (也可改成動態參數或 YAML)
# ────────────────────────────────────────────────────────
FORWARD_V        = 0.25      # m/s  直行速度
TURN_LIN_V       = 0.15      # m/s  轉彎時仍保持線速
TURN_ANG_V       = 0.6       # rad/s
BACKWARD_V       = -0.10     # m/s  倒退
BACK_TIME        = 0.5       # s
TURN_TIME        = 0.8       # s
OBST_THRESH      = 0.50      # m 觸發避障距離
LIDAR_ARC_DEG    = 30        # 只看 ±LIDAR_ARC_DEG 正前方
PATROL_DURATION  = 120       # s  巡邏多久後返航
FUSE_Y_LIMIT     = 0.20      # 深度點雲 |y| < 此值才算正前
RETURN_TOL       = 0.20      # m  離起點小於此值視為「到家」
HZ               = 10        # 主迴圈頻率 (Hz)
# --------------------------------------------------------

class PatrolNode:
    FWD, AVOID, HOME = range(3)

    def __init__(self):
        rospy.init_node('limo_patrol')

        # 參數可用 rosparam 覆蓋
        self.scan_topic  = rospy.get_param('~scan_topic',  '/limo/scan')
        self.cloud_topic = rospy.get_param('~cloud_topic', '/camera/depth/points')

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        rospy.Subscriber(self.scan_topic,  LaserScan,   self.scan_cb,  queue_size=1)
        rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_cb, queue_size=1)
        rospy.Subscriber('/odom', Odometry, self.odom_cb, queue_size=1)

        self.state = self.FWD
        self.state_ts = rospy.Time.now()
        self.obstacle_ahead = False

        # odom tracking
        self.odom_ready, self.x0, self.y0 = False, 0.0, 0.0
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0

        self.start_time = time.time()

        rospy.loginfo("巡邏節點啟動：%.0fs 後自動返航", PATROL_DURATION)
        self.main_loop()

    # ──────────── callback 區 ────────────
    def scan_cb(self, msg: LaserScan):
        # 取 ±LIDAR_ARC_DEG 範圍最小距離
        total_samples = len(msg.ranges)
        deg_per_sample = 360.0 / total_samples
        half_arc_samples = int(LIDAR_ARC_DEG / deg_per_sample)
        center = total_samples // 2                     # 0° 在後半段；前方是中間
        front_arc = msg.ranges[center - half_arc_samples : center + half_arc_samples + 1]
        d_lidar = min([r for r in front_arc if not math.isinf(r)] + [np.inf])
        self.obstacle_ahead = (d_lidar < OBST_THRESH)

    def cloud_cb(self, msg: PointCloud2):
        # 取 |y|<FUSE_Y_LIMIT 且 z<2m 的最近點
        d_cloud = np.inf
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p
            if abs(y) < FUSE_Y_LIMIT and 0.0 < x < OBST_THRESH and z < 2.0:
                d_cloud = min(d_cloud, x)
        # 只要雲也偵測到近障礙，就強制避障
        if d_cloud < OBST_THRESH:
            self.obstacle_ahead = True

    def odom_cb(self, msg: Odometry):
        pose = msg.pose.pose
        self.x = pose.position.x
        self.y = pose.position.y
        ori = pose.orientation
        _, _, self.yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        if not self.odom_ready:
            self.x0, self.y0 = self.x, self.y
            self.odom_ready = True

    # ──────────── 動作 helpers ────────────
    def publish_cmd(self, lin, ang):
        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
        self.cmd_pub.publish(cmd)

    def stop(self):
        self.publish_cmd(0, 0)

    # ───────── 主迴圈 ─────────
    def main_loop(self):
        rate = rospy.Rate(HZ)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - self.state_ts).to_sec()

            # 巡邏時長到 → 切 HOME
            if self.state != self.HOME and time.time() - self.start_time > PATROL_DURATION:
                rospy.loginfo("巡邏時間到，返航中…")
                self.state = self.HOME
                self.state_ts = now

            # 狀態機
            if self.state == self.FWD:
                if self.obstacle_ahead:
                    self.state = self.AVOID
                    self.state_ts = now
                    rospy.logdebug("→ AVOID")
                else:
                    self.publish_cmd(FORWARD_V, 0)

            elif self.state == self.AVOID:
                if dt < BACK_TIME:
                    self.publish_cmd(BACKWARD_V, 0)                 # 後退
                elif dt < BACK_TIME + TURN_TIME:
                    self.publish_cmd(TURN_LIN_V,  TURN_ANG_V)       # 左轉繞
                else:
                    # 結束避障
                    self.state = self.FWD
                    self.state_ts = now
                    rospy.logdebug("→ FWD")

            elif self.state == self.HOME:
                if not self.odom_ready:
                    self.stop()
                else:
                    dx = self.x0 - self.x
                    dy = self.y0 - self.y
                    dist_home = math.hypot(dx, dy)
                    if dist_home < RETURN_TOL:
                        rospy.loginfo("已返抵起點，停止")
                        self.stop()
                        break

                    target_yaw = math.atan2(dy, dx)
                    yaw_err = self.angle_diff(target_yaw, self.yaw)

                    # 同時轉向＋前進
                    lin = max(0.10, FORWARD_V * (dist_home / 2.0))   # 越近越慢
                    ang = max(min(yaw_err, 1.0), -1.0)               # 限制 ±1 rad/s
                    self.publish_cmd(lin, ang)

            self.obstacle_ahead = False      # 本週期檢查完畢
            rate.sleep()

    @staticmethod
    def angle_diff(a, b):
        d = a - b
        while d > math.pi:  d -= 2*math.pi
        while d < -math.pi: d += 2*math.pi
        return d

# ────── main ──────
if __name__ == '__main__':
    try:
        PatrolNode()
    except rospy.ROSInterruptException:
        pass

