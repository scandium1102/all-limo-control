#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
limo_patrol.py ── LIMO 主控節點 (短期版)
--------------------------------------
依序行為：
  FWD(牆隨) → AVOID(backoff) ↻ → RECOVERY(stuck) ↻ → RETURN → STOP
"""

import rospy
from geometry_msgs.msg import Twist

# 匯入模組
from patrol_modules import (
    SensorHub,
    WallFollower,
    BackoffBehavior,
    StuckRecovery,
    ReturnHome,
    CoverageGrid,
)

# --------------------------------------------------
class LimoPatrolMain:
    def __init__(self):
        rospy.init_node("limo_patrol", anonymous=False)

        # ---------- 參數 ----------
        self.rate_hz = rospy.get_param("~rate_hz", 20)
        self.cover_goal = rospy.get_param("~coverage_ratio_goal", 1.0)
        self.patrol_dur = float(rospy.get_param("~patrol_duration", 0.0))

        # ---------- 物件 ----------
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.sensor = SensorHub()
        self.grid = CoverageGrid(rospy.get_param("~coverage_cell_size", 0.5))
        self.wf = WallFollower(self.sensor)
        self.backoff = BackoffBehavior(self.sensor, self.cmd_pub)
        self.recovery = StuckRecovery(self.sensor, self.cmd_pub)
        self.returner = None  # 返航物件啟動時建立

        # ---------- 狀態 ----------
        self.state = "INIT"  # INIT → FWD → AVOID / RECOVERY → RETURN → STOP
        self.start_time = rospy.Time.now()
        self.home_pos = None

    # --------------------------------------------------
    def loop(self):
        rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("等待 SensorHub 就緒...")
        while not rospy.is_shutdown() and not self.sensor.ready:
            rate.sleep()
        rospy.loginfo("SensorHub 就緒，開始巡邏")

        # 記錄起點
        self.home_pos = self.sensor.pose()[:2]

        while not rospy.is_shutdown():
            if self.state == "FWD":
                self.do_fwd()
            elif self.state == "AVOID":
                self.do_avoid()
            elif self.state == "RECOVERY":
                self.do_recovery()
            elif self.state == "RETURN":
                self.do_return()
            elif self.state == "STOP":
                self.cmd_pub.publish(Twist())
                rospy.loginfo_throttle(5, "巡邏任務完成，已停止。")
                rate.sleep()
                continue
            elif self.state == "INIT":
                # 啟動牆隨
                self.state = "FWD"

            # 每圈更新覆蓋率 & 里程計
            x, y, _ = self.sensor.pose()
            if x is not None:
                self.grid.mark(x, y)

            # 巡邏完成條件
            if self.state == "FWD":
                if 0 < self.patrol_dur <= (rospy.Time.now() - self.start_time).to_sec():
                    rospy.loginfo("巡邏時長達 %.1fs → 返航", self.patrol_dur)
                    self.start_return()
                elif self.cover_goal_met():
                    rospy.loginfo("覆蓋率 %.0f%% 達標 → 返航", self.grid.percent())
                    self.start_return()

            rate.sleep()

    # --------------------------------------------------
    # 行為實作
    # --------------------------------------------------
    def do_fwd(self):
        cmd, blocked = self.wf.compute_cmd()
        if blocked:
            # 切換到避障
            self.backoff.start(turn_dir="left")
            self.state = "AVOID"
            rospy.loginfo("FWD→AVOID")
            return

        # 卡死監聽
        stuck, _ = self.recovery.monitor()
        if stuck:
            self.state = "RECOVERY"
            rospy.loginfo("FWD→RECOVERY")
            return

        self.cmd_pub.publish(cmd)

    # --------------------------------------------------
    def do_avoid(self):
        self.backoff.step()
        if self.backoff.done:
            self.state = "FWD"
            rospy.loginfo("AVOID→FWD")

    # --------------------------------------------------
    def do_recovery(self):
        # monitor() 已標示 in_recovery
        self.recovery.execute()
        if self.recovery.done:
            self.state = "FWD"
            rospy.loginfo("RECOVERY→FWD")

    # --------------------------------------------------
    def start_return(self):
        self.returner = ReturnHome(self.sensor, self.cmd_pub, self.home_pos)
        self.state = "RETURN"
        rospy.loginfo("進入 RETURN 模式")

    def do_return(self):
        done, blocked = self.returner.step()
        if blocked:
            self.backoff.start(turn_dir="left")
            self.state = "AVOID"
            rospy.loginfo("RETURN→AVOID")
        elif done:
            self.state = "STOP"

    # --------------------------------------------------
    def cover_goal_met(self):
        if self.cover_goal >= 1.0:
            return False
        return self.grid.ratio() >= self.cover_goal


# --------------------------------------------------
if __name__ == "__main__":
    try:
        LimoPatrolMain().loop()
    except rospy.ROSInterruptException:
        pass

