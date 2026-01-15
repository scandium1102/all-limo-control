# LIMO 專題專案 README（Gazebo 分支）

這份 README 給學弟快速了解我們在 LIMO 機器人上做了哪些事、目前進度、遇到的困難與卡點，以及如何在模擬/實機啟動。

## 這個 repo 有包含實體 LIMO 的所有程式碼嗎？
**不完全**。本 repo 主要是我們自行開發的 `limo_control`（探索、避障、房間分割、房間導航、NLU 介接等）。
實體 LIMO 的驅動與感測器套件（例如 `limo_bringup`、`astra_camera`）通常是官方/外部套件，不在這個 repo 內，需要另外安裝。

## 我們做了什麼（核心功能）
- **SLAM 建圖**：`slam_toolbox` 產生 /map。
- **前緣探索**：`explore_lite` 產生目標 → `move_base` 規劃。
- **安全濾波**：`cmd_vel_safety_filter.py` 以 LiDAR 距離/TTC 保險減速/停車。
- **避障兜底**：`lidar_avoidance_node.py` + `cmd_vel_arbiter.py`，在導航失敗時接管。
- **探索完成監督**：`explore_supervisor.py` 判定完成 → 回家。
- **房間分割/標記**：`room_segmenter.py`，產生 `room_1/room_2...`。
- **房間導航**：`room_navigator.py`，服務 `/room_navigator/go_to_room`。
- **NLU 介面**：`gemini_nlu.py`，用 Google Gemini API 將指令轉 JSON。

## 重要文件（建議先看）
- `GAZEBO_SETUP_LOG.md`：Gazebo/ROS 環境準備紀錄。
- `docs/VLN_PROGRESS.md`：目前進度與下一步。
- `docs/SESSION_LOG.md`：近期更新紀錄。
- `docs/LEGACY_UNUSED.md`：舊版/封存內容說明。

## 快速啟動（模擬 Gazebo）
1) 世界/感測器：
```bash
roslaunch limo_control limo_sim_sensors.launch gui:=true world:=<your_world.world>
```
2) SLAM：
```bash
roslaunch limo_control slam.launch
```
3) 前緣探索（含安全/避障/仲裁/監督）：
```bash
roslaunch limo_control explore_lite.launch
```
4) 房間分割與標記（可選）：
```bash
roslaunch limo_control room_segmentation.launch output_yaml:=~/maps/rooms.yaml
```
5) 房間導航服務（可選）：
```bash
roslaunch limo_control room_nav.launch rooms_yaml:=~/maps/rooms.yaml
```
6) NLU（Gemini，可選）：
```bash
pip install google-generativeai
export GOOGLE_API_KEY=<你的key>
roslaunch limo_control gemini_nlu.launch
```
7) 前往房間範例：
```bash
rosservice call /room_navigator/go_to_room "name: 'room_1' dx: 0 dy: 0 yaw: 0"
```

## 實機啟動（重要）
以下指令為實體 LIMO 啟動常用流程（外部套件需先安裝）：

- 啟動相機（Astra）：
```bash
roslaunch astra_camera dabai_dc1.launch
```
- 啟動底盤/雷射（LIMO 官方 bringup）：
```bash
roslaunch limo_bringup limo_start.launch
```

啟動完成後，再跑我們的 `slam.launch`、`explore_lite.launch`、`room_segmenter` 等節點即可。

## 工作流與套件串接（簡述）
- **explore_lite** 產生前緣目標 → **move_base** 規劃 `/cmd_vel_raw`
- **cmd_vel_safety_filter.py** 以 /scan 做安全濾波 → `/cmd_vel_nav`
- **lidar_avoidance_node.py** 產生 `/cmd_vel_avoid`（保險）
- **cmd_vel_arbiter.py** 在導航失敗/超時時切換至避障 → `/cmd_vel`
- **explore_supervisor.py** 監控未知比例與 idle，完成後送回家目標
- **room_segmenter.py** 從 /map 產生房間標記
- **room_navigator.py** 服務導引到特定房間
- **gemini_nlu.py** 使用 Gemini API 將自然語言轉 JSON（供任務管理器使用）

## 遇到的困難與卡點（給學弟參考）
- **牆面誤判動態**：車在移動時牆變成「動態物體」→ WAIT 觸發，需用 ego speed 補償或提高門檻。
- **costmap/安全濾波過嚴**：過度膨脹導致車不轉彎直接停；需調 `inflation_radius`、`stop_distance`。
- **探索停住**：`progress_timeout` 太長/太短會卡住或頻繁丟目標。
- **房間分割穩定性**：門口寬度、未知區域判定會影響分割結果。
- **VLN 仍欠缺視覺語義**：房號/OCR/物件標籤尚未與房間綁定。

## 下一步方向（未完成的關鍵）
- **任務管理器**：解析 NLU JSON → 執行多段任務（去 A 再去 B 回家）。
- **視覺語義**：OCR/ArUco/YOLO 讓「房間名稱」可視覺辨識並綁定地圖。
- **實機調參**：動態障礙、門口通行性、local planner 參數需要在真機微調。

---
若學弟接手，建議先看 `docs/VLN_PROGRESS.md`，並在 Gazebo 重跑一次完整流程確認環境正常。
有任何疑問可以先從 `GAZEBO_SETUP_LOG.md` 對照安裝與設定。
