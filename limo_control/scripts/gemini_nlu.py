#!/usr/bin/env python3
"""
gemini_nlu.py

ROS service wrapper around Google Gemini (google-generativeai) to parse
natural-language commands into a JSON-ish string for downstream mission manager.
Requires environment variable GOOGLE_API_KEY.

Service:
  /gemini_nlu/parse (limo_control/ParseText)
Params:
  ~model (default: gemini-1.5-flash)
  ~temperature (default: 0.2)
  ~request_timeout (default: 15.0) seconds
  ~prompt_template (optional string to prepend)
"""

from __future__ import annotations

import os
import rospy
from limo_control.srv import ParseText, ParseTextResponse

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class GeminiNLU:
    def __init__(self):
        self.model_name = rospy.get_param("~model", "gemini-1.5-flash")
        self.temperature = float(rospy.get_param("~temperature", 0.2))
        self.request_timeout = float(rospy.get_param("~request_timeout", 15.0))
        self.prompt_template = rospy.get_param(
            "~prompt_template",
            "請把以下指令轉成 JSON 結構，包含 intent、rooms(list)、actions(list)，輸出純 JSON：\n",
        )

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if genai is None:
            rospy.logwarn("google-generativeai not installed; pip install google-generativeai to enable Gemini NLU.")
        elif not api_key:
            rospy.logwarn("GOOGLE_API_KEY not set; Gemini NLU will reject requests.")
        else:
            genai.configure(api_key=api_key)
            rospy.loginfo("Gemini NLU configured with model %s", self.model_name)

        self._srv = rospy.Service("~parse", ParseText, self._on_parse)

    def _on_parse(self, req: ParseText.Request) -> ParseTextResponse:
        resp = ParseTextResponse(success=False, json_cmd="", error="")
        if genai is None:
            resp.error = "google-generativeai not installed"
            return resp
        if not os.environ.get("GOOGLE_API_KEY", ""):
            resp.error = "GOOGLE_API_KEY not set"
            return resp
        text = req.text.strip()
        if not text:
            resp.error = "empty text"
            return resp

        prompt = f"{self.prompt_template}{text}"
        try:
            model = genai.GenerativeModel(self.model_name)
            result = model.generate_content(prompt, generation_config={"temperature": self.temperature}, request_options={"timeout": self.request_timeout})
            content = result.text or ""
            content = content.strip()
            if not content:
                resp.error = "empty response"
                return resp
            resp.success = True
            resp.json_cmd = content
            return resp
        except Exception as exc:
            resp.error = f"Gemini error: {exc}"
            return resp


def main():
    rospy.init_node("gemini_nlu")
    GeminiNLU()
    rospy.loginfo("gemini_nlu node started")
    rospy.spin()


if __name__ == "__main__":
    main()
