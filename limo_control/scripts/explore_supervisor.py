#!/usr/bin/env python3
"""
explore_supervisor.py

Lightweight supervisor for explore_lite + move_base:
- Tracks map unknown ratio (low-pass).
- Detects when exploration is idle (no active move_base goals) and map is mostly complete,
  then publishes a home goal to /move_base_simple/goal.
- Publishes simple coverage milestones as RViz text markers when unknown ratio drops past steps.
"""

import math
import threading
from typing import Optional

import numpy as np
import rospy
import tf2_ros
from actionlib_msgs.msg import GoalStatusArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from visualization_msgs.msg import Marker


class ExploreSupervisor:
    def __init__(self):
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.status_topic = rospy.get_param("~status_topic", "/move_base/status")
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.complete_unknown = rospy.get_param("~complete_unknown", 0.05)
        self.unknown_lpf_alpha = rospy.get_param("~unknown_lpf_alpha", 0.3)
        self.idle_time = rospy.get_param("~idle_time", 5.0)
        self.milestone_step = rospy.get_param("~milestone_step", 0.1)  # 10% steps
        self.marker_lifetime = rospy.get_param("~marker_lifetime", 0.0)

        self._tf = tf2_ros.Buffer()
        self._tfl = tf2_ros.TransformListener(self._tf)
        self._lock = threading.RLock()

        self._unknown_ratio: Optional[float] = None
        self._home_pose: Optional[PoseStamped] = None
        self._last_active_time: Optional[rospy.Time] = None
        self._home_sent: bool = False
        self._next_milestone: float = 1.0 - self.milestone_step
        self._marker_seq: int = 0

        rospy.Subscriber(self.map_topic, OccupancyGrid, self._map_cb, queue_size=1)
        rospy.Subscriber(self.status_topic, GoalStatusArray, self._status_cb, queue_size=5)

        self._goal_pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1, latch=False)
        self._marker_pub = rospy.Publisher("~coverage_markers", Marker, queue_size=10, latch=True)

        rospy.Timer(rospy.Duration(0.5), self._timer_cb)

    def _map_cb(self, msg: OccupancyGrid):
        with self._lock:
            data = np.asarray(msg.data, dtype=np.int8)
            total = data.size
            if total == 0:
                return
            unknown = float(np.count_nonzero(data == -1))
            ratio = unknown / float(total)
            if self._unknown_ratio is None:
                self._unknown_ratio = ratio
            else:
                a = max(0.0, min(1.0, self.unknown_lpf_alpha))
                self._unknown_ratio = a * ratio + (1.0 - a) * self._unknown_ratio

            # Store home pose once map is available
            if self._home_pose is None:
                pose = self._lookup_pose()
                if pose is not None:
                    self._home_pose = pose

    def _status_cb(self, msg: GoalStatusArray):
        now = rospy.Time.now()
        active = any(st.status == 1 for st in msg.status_list)  # ACTIVE=1
        if active:
            self._last_active_time = now

    def _lookup_pose(self) -> Optional[PoseStamped]:
        try:
            trans = self._tf.lookup_transform(self.map_frame, self.base_frame, rospy.Time(0), rospy.Duration(0.2))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            return None
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = self.map_frame
        ps.pose.position.x = trans.transform.translation.x
        ps.pose.position.y = trans.transform.translation.y
        ps.pose.position.z = 0.0
        ps.pose.orientation = trans.transform.rotation
        return ps

    def _timer_cb(self, _event):
        with self._lock:
            now = rospy.Time.now()

            # Coverage milestone markers
            if self._unknown_ratio is not None and self._unknown_ratio <= self._next_milestone:
                pose = self._lookup_pose()
                if pose is not None:
                    self._publish_marker(pose, self._unknown_ratio)
                self._next_milestone = max(0.0, self._next_milestone - self.milestone_step)

            # Decide if idle and complete
            idle = False
            if self._last_active_time is None:
                idle = True
            else:
                idle = (now - self._last_active_time).to_sec() >= self.idle_time

            complete = self._unknown_ratio is not None and self._unknown_ratio <= self.complete_unknown

            if complete and idle and not self._home_sent:
                if self._home_pose is None:
                    self._home_pose = self._lookup_pose()
                if self._home_pose is not None:
                    self._home_pose.header.stamp = rospy.Time.now()
                    self._goal_pub.publish(self._home_pose)
                    rospy.loginfo("ExploreSupervisor: exploration complete, sending home goal.")
                    self._home_sent = True

    def _publish_marker(self, pose: PoseStamped, unknown_ratio: float):
        self._marker_seq += 1
        m = Marker()
        m.header = Header(frame_id=self.map_frame, stamp=rospy.Time.now())
        m.ns = "coverage"
        m.id = self._marker_seq
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose = pose.pose
        m.pose.position.z += 0.4
        pct = 100.0 * (1.0 - unknown_ratio)
        m.text = f"Coverage {pct:.1f}%"
        m.scale.z = 0.25
        m.color.r = 0.1
        m.color.g = 0.8
        m.color.b = 0.1
        m.color.a = 1.0
        m.lifetime = rospy.Duration(self.marker_lifetime)
        self._marker_pub.publish(m)


def main():
    rospy.init_node("explore_supervisor")
    ExploreSupervisor()
    rospy.loginfo("explore_supervisor started")
    rospy.spin()


if __name__ == "__main__":
    main()
