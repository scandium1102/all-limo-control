#!/usr/bin/env python3
"""
room_navigator.py

Service-based navigator that accepts a room name and sends a MoveBase goal to
the room centroid. Room sources:
- YAML file (from room_segmenter)
- live markers on /rooms/markers (first line of text is name)
"""

from __future__ import annotations

import math
import os
import threading
from typing import Dict, Optional, Tuple

import actionlib
import rospy
import yaml
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from limo_control.srv import GoToRoom, GoToRoomResponse
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from visualization_msgs.msg import MarkerArray


class RoomNavigator:
    def __init__(self):
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.rooms_yaml = rospy.get_param("~rooms_yaml", "")
        self.goal_timeout = float(rospy.get_param("~goal_timeout", 120.0))
        self.wait_for_server = float(rospy.get_param("~wait_for_server", 10.0))

        self._lock = threading.RLock()
        self._rooms: Dict[str, Tuple[float, float, float]] = {}  # name -> (x, y, yaw)

        if self.rooms_yaml:
            self._load_yaml(self.rooms_yaml)

        rospy.Subscriber("rooms/markers", MarkerArray, self._markers_cb, queue_size=1)

        self._client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        if not self._client.wait_for_server(rospy.Duration(self.wait_for_server)):
            rospy.logwarn("room_navigator: move_base server not available after %.1fs", self.wait_for_server)

        self._srv = rospy.Service("~go_to_room", GoToRoom, self._on_go_to_room)

    # ------------------------------------------------------------
    def _load_yaml(self, path: str):
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            rospy.logwarn("room_navigator: rooms_yaml not found: %s", path)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:
            rospy.logwarn("room_navigator: failed to load %s: %s", path, exc)
            return
        rooms = data.get("rooms", [])
        with self._lock:
            for r in rooms:
                name = r.get("name")
                c = r.get("centroid", {})
                if name is None or "x" not in c or "y" not in c:
                    continue
                self._rooms[name] = (float(c["x"]), float(c["y"]), 0.0)
        rospy.loginfo("room_navigator: loaded %d rooms from %s", len(self._rooms), path)

    def _markers_cb(self, msg: MarkerArray):
        updated = 0
        with self._lock:
            for m in msg.markers:
                if not m.text:
                    continue
                name = m.text.splitlines()[0].strip()
                if not name:
                    continue
                self._rooms[name] = (m.pose.position.x, m.pose.position.y, 0.0)
                updated += 1
        if updated:
            rospy.logdebug("room_navigator: updated %d rooms from markers", updated)

    def _on_go_to_room(self, req: GoToRoom.Request) -> GoToRoomResponse:
        resp = GoToRoomResponse(success=False, message="")
        with self._lock:
            target = self._rooms.get(req.name)
        if target is None:
            resp.message = f"room '{req.name}' not found"
            return resp
        x, y, yaw = target
        x += req.dx
        y += req.dy
        yaw_cmd = req.yaw if abs(req.yaw) > 1e-3 else yaw

        goal = MoveBaseGoal()
        goal.target_pose = PoseStamped()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = math.sin(yaw_cmd / 2.0)
        goal.target_pose.pose.orientation.w = math.cos(yaw_cmd / 2.0)

        if not self._client.wait_for_server(rospy.Duration(self.wait_for_server)):
            resp.message = "move_base not available"
            return resp

        self._client.send_goal(goal)
        finished = self._client.wait_for_result(rospy.Duration(self.goal_timeout))
        if not finished:
            self._client.cancel_goal()
            resp.message = "goal timeout"
            return resp
        status = self._client.get_state()
        if status == GoalStatus.SUCCEEDED:
            resp.success = True
            resp.message = "reached"
        else:
            resp.success = False
            resp.message = f"move_base status {status}"
        return resp


def main():
    rospy.init_node("room_navigator")
    RoomNavigator()
    rospy.loginfo("room_navigator started")
    rospy.spin()


if __name__ == "__main__":
    main()
