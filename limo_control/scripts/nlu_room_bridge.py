#!/usr/bin/env python3
"""
nlu_room_bridge.py

Bridge natural-language text to room navigation:
1) Subscribes to std_msgs/String on ~command (default: /vln_command).
2) Calls Gemini NLU service (~parse) to get JSON.
3) Parses intent/room/offset, then calls GoToRoom service to dispatch a goal.

Expected JSON (flexible, best effort):
{
  "intent": "goto",
  "room": "kitchen" | or "rooms": ["kitchen", ...],
  "dx": 0.0, "dy": 0.0, "yaw": 0.0
}
"""

from __future__ import annotations

import json
import rospy
from std_msgs.msg import String
from limo_control.srv import ParseText, ParseTextRequest, GoToRoom, GoToRoomRequest


class NluRoomBridge:
    def __init__(self):
        self.nlu_service = rospy.get_param("~nlu_service", "/gemini_nlu/parse")
        self.goto_service = rospy.get_param("~goto_service", "/room_navigator/go_to_room")
        self.required_intent = rospy.get_param("~required_intent", "goto")

        self._nlu = rospy.ServiceProxy(self.nlu_service, ParseText)
        self._goto = rospy.ServiceProxy(self.goto_service, GoToRoom)

        self.sub = rospy.Subscriber("~command", String, self._on_cmd, queue_size=10)
        rospy.loginfo("nlu_room_bridge listening on %s, calling NLU=%s, GoToRoom=%s",
                      self.sub.name, self.nlu_service, self.goto_service)

    def _on_cmd(self, msg: String):
        text = msg.data.strip()
        if not text:
            rospy.logwarn("nlu_room_bridge: empty command")
            return
        try:
            nlu_req = ParseTextRequest(text=text)
            nlu_resp = self._nlu(nlu_req)
        except Exception as exc:
            rospy.logerr("nlu_room_bridge: NLU call failed: %s", exc)
            return
        if not nlu_resp.success:
            rospy.logwarn("nlu_room_bridge: NLU error: %s", nlu_resp.error)
            return

        payload = self._parse_json(nlu_resp.json_cmd)
        if payload is None:
            rospy.logwarn("nlu_room_bridge: failed to parse JSON from NLU: %s", nlu_resp.json_cmd)
            return

        intent = str(payload.get("intent", "")).lower()
        if self.required_intent and intent != self.required_intent:
            rospy.logwarn("nlu_room_bridge: intent '%s' != required '%s'", intent, self.required_intent)
            return

        room = payload.get("room", "")
        if not room:
            rooms = payload.get("rooms", [])
            if isinstance(rooms, list) and rooms:
                room = str(rooms[0])
        if not room:
            rospy.logwarn("nlu_room_bridge: no room found in payload: %s", payload)
            return

        dx = float(payload.get("dx", 0.0))
        dy = float(payload.get("dy", 0.0))
        yaw = float(payload.get("yaw", 0.0))

        try:
            req = GoToRoomRequest(name=room, dx=dx, dy=dy, yaw=yaw)
            resp = self._goto(req)
            if resp.success:
                rospy.loginfo("nlu_room_bridge: dispatched to room '%s' (dx=%.2f,dy=%.2f,yaw=%.2f)",
                              room, dx, dy, yaw)
            else:
                rospy.logwarn("nlu_room_bridge: GoToRoom failed: %s", resp.message)
        except Exception as exc:
            rospy.logerr("nlu_room_bridge: GoToRoom call error: %s", exc)

    def _parse_json(self, txt: str):
        try:
            return json.loads(txt)
        except Exception:
            # Try to strip code fences
            cleaned = txt.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
            try:
                return json.loads(cleaned)
            except Exception:
                return None


def main():
    rospy.init_node("nlu_room_bridge")
    NluRoomBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
