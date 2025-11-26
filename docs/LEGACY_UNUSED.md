Legacy / unused nodes (kept for reference; do not use in the current pipeline unless explicitly enabled)

- `limo_control/scripts/limo_patrol.py` (legacy patrol FSM)
- `limo_control/scripts/limo_patrol_v6.py` (legacy patrol v6)
- Any launches that start these nodes are considered deprecated.

Notes:
- These scripts now require `~allow_legacy:=true` to run; otherwise they exit immediately.
- Current pipeline uses `explore_lite` + `move_base` + safety/arbiter + lidar_avoid, plus room navigation/NLU (gemini_nlu + nlu_room_bridge).
