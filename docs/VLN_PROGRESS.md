## VLN Progress Log (current branch: Gazebo)

### Implemented
- Map/build/explore: slam_toolbox + explore_lite + move_base with safety_filter + arbiter (lidar_avoid fallback), supervisor (complete -> home).
- Room segmentation: `room_segmenter.py` + `room_segmentation.launch` outputs `/rooms/centers`, `/rooms/markers`, optional YAML, `/room_segmenter/save`.
- Room navigation: `room_navigator.py` + `room_nav.launch`, service `/room_navigator/go_to_room` (GoToRoom.srv), uses room centroids from YAML or `/rooms/markers` → sends move_base goal (still goes through safety/arbiter).
- Gemini NLU stub: `gemini_nlu.py` + `gemini_nlu.launch`, service `/gemini_nlu/parse` (ParseText.srv); calls Google Gemini via `google-generativeai` if `GOOGLE_API_KEY` is set.
- New services: `GoToRoom.srv`, `ParseText.srv`; CMake/package updated with actionlib/move_base_msgs/message_generation.

### How to run (sim flow)
1) Sensors/world: `roslaunch limo_control limo_sim_sensors.launch gui:=true world:=<your_world.world>`
2) SLAM: `roslaunch limo_control slam.launch`
3) Explore stack: `roslaunch limo_control explore_lite.launch`
4) Room segmentation: `roslaunch limo_control room_segmentation.launch output_yaml:=~/maps/rooms.yaml`
5) Room navigation: `roslaunch limo_control room_nav.launch rooms_yaml:=~/maps/rooms.yaml`
6) (Optional) Gemini NLU: `roslaunch limo_control gemini_nlu.launch` (requires `pip install google-generativeai` and `GOOGLE_API_KEY`)
7) Send room goal: `rosservice call /room_navigator/go_to_room "name: 'room_1' dx: 0 dy: 0 yaw: 0"`
8) NLU demo: `rosservice call /gemini_nlu/parse "text: '去小房間1 繞一圈再回來'"` → returns JSON-ish string.

### TODO / Next steps
- Mission manager: consume ParseText output, queue multi-room tasks, support actions (loop/photograph), retries/cancel.
- Visual grounding: add OCR/ArUco/YOLO to bind doorplates/objects to rooms; update room YAML/markers.
- Hardening: handle move_base failures better, add tests/rostests for services, document required pip (`google-generativeai`).
- Real robot launch: add real-world bringup that reuses the same cmd_vel pipeline and room services.

### Notes
- Gemini is cloud-only; no local GPU needed. Set `GOOGLE_API_KEY` env var. The node degrades gracefully if key/module missing.
- Untracked files in patrol_modules remain uncommitted (coverage_grid.py, sensors.py, utils.py, wall_follow.py).
