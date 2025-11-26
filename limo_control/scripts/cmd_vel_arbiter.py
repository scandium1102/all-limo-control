#!/usr/bin/env python3
"""
cmd_vel_arbiter.py

Select between navigation cmd_vel (from move_base/explore_lite with safety filter)
and lidar_avoidance fallback commands. When move_base reports failure (ABORTED /
REJECTED / LOST) or nav cmd timeout occurs, the arbiter switches to lidar_avoid
for a short recovery window, then returns to nav.
"""

import rospy
from geometry_msgs.msg import Twist
from actionlib_msgs.msg import GoalStatusArray


class CmdVelArbiter:
    def __init__(self):
        self.recovery_time = rospy.get_param("~recovery_time", 3.0)
        self.nav_timeout = rospy.get_param("~nav_timeout", 1.0)

        self.nav_cmd = Twist()
        self.avoid_cmd = Twist()
        self.last_nav_stamp = rospy.Time(0)
        self.recovery_until = rospy.Time(0)

        self.sub_nav = rospy.Subscriber("cmd_nav", Twist, self._nav_cb, queue_size=10)
        self.sub_avoid = rospy.Subscriber("cmd_avoid", Twist, self._avoid_cb, queue_size=10)
        self.sub_status = rospy.Subscriber("move_base/status", GoalStatusArray, self._status_cb, queue_size=5)

        self.pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(0.05), self._on_timer)

    def _nav_cb(self, msg: Twist):
        self.nav_cmd = msg
        self.last_nav_stamp = rospy.Time.now()

    def _avoid_cb(self, msg: Twist):
        self.avoid_cmd = msg

    def _status_cb(self, msg: GoalStatusArray):
        # If move_base reports failure, enter recovery window
        for st in msg.status_list:
            if st.status in (4, 5, 9):  # ABORTED=4, REJECTED=5, LOST=9
                self.recovery_until = rospy.Time.now() + rospy.Duration(self.recovery_time)
                break

    def _on_timer(self, _):
        now = rospy.Time.now()
        use_recovery = now < self.recovery_until

        # Also treat nav timeout as trigger
        if not use_recovery and (now - self.last_nav_stamp).to_sec() > self.nav_timeout:
            use_recovery = True
            self.recovery_until = now + rospy.Duration(self.recovery_time)

        cmd = self.avoid_cmd if use_recovery else self.nav_cmd
        self.pub.publish(cmd)


def main():
    rospy.init_node("cmd_vel_arbiter")
    CmdVelArbiter()
    rospy.loginfo("cmd_vel_arbiter started")
    rospy.spin()


if __name__ == "__main__":
    main()
