#!/usr/bin/env python3
"""Depth-enhanced reactive avoidance node for the LIMO platform."""

from __future__ import annotations

import json
import math
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import ros_numpy
import rospy
import tf2_ros
from diagnostic_updater import FunctionDiagnosticTask, Updater
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
from tf.transformations import quaternion_matrix
from visualization_msgs.msg import Marker, MarkerArray

import rospkg
import message_filters
from dynamic_reconfigure.server import Server as DynServer

try:
    from limo_control.cfg import AvoidanceConfig, DepthHazardConfig
except ImportError:
    AvoidanceConfig = None
    DepthHazardConfig = None
    rospy.logwarn(
        "[depth_avoidance] limo_control.cfg modules not found; dynamic_reconfigure disabled"
    )

from patrol_modules.dynamic_tracker import DynamicTracker, TrackParams
from patrol_modules.lidar_avoid import AvoidParams, LidarAvoider


@dataclass
class HazardReport:
    """3D hazard summary derived from the depth point cloud."""

    speed_scale: float = 1.0
    slope_angle_deg: float = 0.0
    slope_blocked: bool = False
    drop_detected: bool = False
    drop_depth: float = 0.0
    overhead_detected: bool = False
    overhead_clearance: float = 0.0
    lateral_bias: float = 0.0
    notes: List[str] = field(default_factory=list)
    sample_count: int = 0

    def as_dict(self) -> Dict:
        return {
            "speed_scale": self.speed_scale,
            "slope_angle_deg": self.slope_angle_deg,
            "slope_blocked": self.slope_blocked,
            "drop_detected": self.drop_detected,
            "drop_depth": self.drop_depth,
            "overhead_detected": self.overhead_detected,
            "overhead_clearance": self.overhead_clearance,
            "lateral_bias": self.lateral_bias,
            "notes": list(self.notes),
            "sample_count": self.sample_count,
        }


@dataclass
class HazardParams:
    slope_stop_deg: float = 12.0
    slope_resume_deg: float = 10.0
    drop_stop_m: float = 0.06
    drop_resume_m: float = 0.03
    overhead_stop_m: float = 0.30
    overhead_resume_m: float = 0.40


class DepthAvoidanceNode:
    """Fuse LiDAR and depth camera data for 3D-aware collision avoidance."""

    def __init__(self) -> None:
        rospy.init_node("depth_avoidance")
        self._lock = threading.RLock()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._avoid_params = self._load_params("lidar_avoidance.yaml", AvoidParams, "avoid_params")
        self._track_params = self._load_params("dynamic_tracker.yaml", TrackParams, "tracker_params")
        self._depth_cfg = self._load_params("depth_avoidance.yaml", dict, "depth")
        self._roi_cfg = self._depth_cfg.get(
            "roi",
            {"x_min": 0.2, "x_max": 3.0, "y_halfwidth": 0.6, "z_min": -0.1, "z_max": 1.8},
        )

        slope_stop = float(self._depth_cfg.get("slope_max_deg", 12.0))
        slope_resume = float(
            self._depth_cfg.get("slope_resume_deg", max(0.0, slope_stop - 2.0))
        )
        if slope_resume >= slope_stop:
            slope_resume = max(0.0, slope_stop - 2.0)

        drop_stop = float(self._depth_cfg.get("drop_height_min", 0.06))
        drop_resume = float(
            self._depth_cfg.get("drop_resume_height", max(0.01, drop_stop * 0.6))
        )
        if drop_resume >= drop_stop:
            drop_resume = max(0.01, drop_stop * 0.6)

        overhead_stop = float(self._depth_cfg.get("overhead_clearance", 0.30))
        overhead_resume = float(self._depth_cfg.get("overhead_resume", overhead_stop + 0.10))
        if overhead_resume <= overhead_stop:
            overhead_resume = overhead_stop + 0.10

        self._hazard_params = HazardParams(
            slope_stop_deg=slope_stop,
            slope_resume_deg=slope_resume,
            drop_stop_m=drop_stop,
            drop_resume_m=drop_resume,
            overhead_stop_m=overhead_stop,
            overhead_resume_m=overhead_resume,
        )

        self._avoider = LidarAvoider(self._avoid_params)
        self._tracker = DynamicTracker(self._track_params)

        self._base_frame = rospy.get_param("~base_frame", "base_link")
        self._depth_frame = rospy.get_param("~depth_frame", "camera_depth_optical_frame")
        self._odom_frame = rospy.get_param("~odom_frame", "odom")

        self._scan_timeout = float(rospy.get_param("~scan_timeout", 0.5))
        self._depth_timeout = float(self._depth_cfg.get("depth_timeout", 0.5))
        self._sync_slop = float(self._depth_cfg.get("sync_slop", 0.12))
        self._min_points = int(self._depth_cfg.get("min_points", 300))
        self._sample_limit = int(self._depth_cfg.get("sample_limit", 60000))
        self._ransac_thresh = float(self._depth_cfg.get("ransac_thresh", 0.02))
        self._spd_alpha = float(self._depth_cfg.get("spd_alpha", 0.30))
        self._look_ahead = float(self._depth_cfg.get("look_ahead", 1.2))
        self._lat_gain_ang = float(self._depth_cfg.get("lat_gain_ang", 0.40))
        self._lat_gain_lin = float(self._depth_cfg.get("lat_gain_lin", 0.20))
        self._hazard_change_limit = float(self._depth_cfg.get("hazard_change_limit", 3.0))
        self._hazard_change_vmax = float(self._depth_cfg.get("hazard_change_vmax", 0.25))

        self._spd_scale_lp = 1.0
        self._lateral_nudge_lp = 0.0
        self._cloud_proc_ms = 0.0
        self._hazard_change_times: deque = deque(maxlen=64)
        self._hazard_signature: Optional[Tuple] = None
        self._hazard_change_rate = 0.0
        self._ats_queue_size = 0
        self._rng = np.random.default_rng()
        self._depth_marker_cache: Optional[MarkerArray] = None

        cmd_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self._cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
        self._debug_pub = rospy.Publisher("~avoidance_debug_3d", String, queue_size=10)
        self._marker_pub = rospy.Publisher("~hazard_markers", MarkerArray, queue_size=1)

        scan_topic = rospy.get_param("~scan_topic", "/scan")
        depth_topic = rospy.get_param("~depth_cloud_topic", "/camera/depth/points_filtered")
        depth_scan_topic = rospy.get_param("~depth_scan_topic", "/camera/depth/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")

        self._scan_sub = message_filters.Subscriber(scan_topic, LaserScan)
        self._depth_scan_sub = message_filters.Subscriber(depth_scan_topic, LaserScan)
        self._depth_sub = message_filters.Subscriber(depth_topic, PointCloud2)
        self._sync: Optional[message_filters.ApproximateTimeSynchronizer]
        self._sync = None
        self._rebuild_sync()

        self._odom_sub = rospy.Subscriber(odom_topic, Odometry, self._odom_cb, queue_size=10)

        self._last_scan_time: Optional[rospy.Time] = None
        self._last_depth_time: Optional[rospy.Time] = None
        self._last_depth_scan_time: Optional[rospy.Time] = None
        self._hazard_state = HazardReport()
        self._last_cmd = Twist()
        self._last_track_count: int = 0
        self._depth_scan_topic = depth_scan_topic
        self._backoff_until = rospy.Time(0)
        self._backoff_dir = 1.0
        self._backoff_speed = 0.12

        self._diag = Updater()
        self._diag.setHardwareID("depth_avoidance")
        self._diag.add(FunctionDiagnosticTask("inputs", self._diag_inputs))

        self._timer = rospy.Timer(rospy.Duration(0.1), self._timer_cb)

        # 將兩組 dynamic_reconfigure 放入不同 namespace，避免 service 名衝突
        if AvoidanceConfig is not None:
            self._dyn_srv = DynServer(AvoidanceConfig, self._on_dyn_cfg, namespace="avoidance")
        else:
            self._dyn_srv = None
        if DepthHazardConfig is not None:
            self._dyn_hazard = DynServer(DepthHazardConfig, self._on_hazard_dyn, namespace="depth")
        else:
            self._dyn_hazard = None

        rospy.on_shutdown(lambda: self._cmd_pub.publish(Twist()))
        rospy.loginfo("Depth avoidance node initialised")

    # ------------------------------------------------------------------
    def _load_params(self, filename: str, cls, param_ns: str):
        pkg_path = rospkg.RosPack().get_path("limo_control")
        default_path = os.path.join(pkg_path, "config", filename)
        param_path = rospy.get_param(f"~{filename}", default_path)

        try:
            with open(param_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except ValueError:
            try:
                with open(param_path, "r", encoding="utf-8") as fh:
                    import yaml

                    data = yaml.safe_load(fh) or {}
            except Exception as exc:  # pragma: no cover - safety log
                rospy.logfatal(f"Failed to load parameters from {param_path}: {exc}")
                raise
        except OSError as exc:
            rospy.logfatal(f"Failed to open {param_path}: {exc}")
            raise

        if isinstance(data, dict):
            if param_ns in data and isinstance(data[param_ns], dict):
                base = dict(data[param_ns])
            else:
                base = dict(data)
        else:
            base = {}

        tree = rospy.get_param(f"~{param_ns}", None)
        if isinstance(tree, dict) and tree:
            override = tree.get(param_ns) if isinstance(tree.get(param_ns), dict) else tree
            if override:
                base.update(override)

        if cls is dict:
            return base
        return cls(**base)

    def _on_dyn_cfg(self, cfg, _level):
        with self._lock:
            self._avoid_params.v_max = float(cfg.v_max)
            self._avoid_params.w_max = float(cfg.w_max)
            k = max(1, int(cfg.median_window))
            if k % 2 == 0:
                k += 1
            self._avoid_params.median_window = k
            self._avoid_params.ttc_stop = float(cfg.ttc_stop)
            self._avoid_params.ttc_slow = float(cfg.ttc_slow)
            cfg.median_window = k
        return cfg

    def _on_hazard_dyn(self, cfg: DepthHazardConfig, _level: int):
        with self._lock:
            self._depth_cfg["slope_max_deg"] = float(cfg.slope_max_deg)
            self._depth_cfg["slope_resume_deg"] = float(cfg.slope_resume_deg)
            self._hazard_params.slope_stop_deg = float(cfg.slope_max_deg)
            resume = float(cfg.slope_resume_deg)
            if resume >= cfg.slope_max_deg:
                resume = max(0.0, cfg.slope_max_deg - 2.0)
            self._hazard_params.slope_resume_deg = resume
            self._depth_cfg["slope_resume_deg"] = resume
            cfg.slope_resume_deg = resume

            self._depth_cfg["drop_height_min"] = float(cfg.drop_height_min)
            self._depth_cfg["drop_resume_height"] = float(cfg.drop_resume_height)
            self._depth_cfg["drop_gap_cells"] = int(cfg.drop_gap_cells)
            self._hazard_params.drop_stop_m = float(cfg.drop_height_min)
            drop_resume = float(
                self._depth_cfg.get("drop_resume_height", max(0.01, cfg.drop_height_min * 0.6))
            )
            if drop_resume >= cfg.drop_height_min:
                drop_resume = max(0.01, cfg.drop_height_min * 0.6)
            self._hazard_params.drop_resume_m = drop_resume
            self._depth_cfg["drop_resume_height"] = drop_resume
            cfg.drop_resume_height = drop_resume

            self._depth_cfg["overhead_clearance"] = float(cfg.overhead_clearance)
            self._depth_cfg["overhead_resume"] = float(cfg.overhead_resume)
            self._hazard_params.overhead_stop_m = float(cfg.overhead_clearance)
            overhead_resume = float(self._depth_cfg.get("overhead_resume", cfg.overhead_clearance + 0.10))
            if overhead_resume <= cfg.overhead_clearance:
                overhead_resume = cfg.overhead_clearance + 0.10
            self._hazard_params.overhead_resume_m = overhead_resume
            self._depth_cfg["overhead_resume"] = overhead_resume
            cfg.overhead_resume = overhead_resume

            self._depth_cfg["look_ahead"] = float(cfg.look_ahead)
            self._depth_cfg["sync_slop"] = float(cfg.sync_slop)
            self._depth_cfg["sample_limit"] = int(cfg.sample_limit)
            self._depth_cfg["ransac_thresh"] = float(cfg.ransac_thresh)
            self._depth_cfg["spd_alpha"] = float(cfg.spd_alpha)
            self._depth_cfg["lat_gain_ang"] = float(cfg.lat_gain_ang)
            self._depth_cfg["lat_gain_lin"] = float(cfg.lat_gain_lin)
            self._depth_cfg["hazard_change_limit"] = float(cfg.hazard_change_limit)
            self._depth_cfg["hazard_change_vmax"] = float(cfg.hazard_change_vmax)
            self._look_ahead = float(cfg.look_ahead)
            old_slop = float(self._sync_slop)
            self._sync_slop = float(cfg.sync_slop)
            self._sample_limit = int(cfg.sample_limit)
            self._ransac_thresh = float(cfg.ransac_thresh)
            self._spd_alpha = float(cfg.spd_alpha)
            self._lat_gain_ang = float(cfg.lat_gain_ang)
            self._lat_gain_lin = float(cfg.lat_gain_lin)
            self._hazard_change_limit = float(cfg.hazard_change_limit)
            self._hazard_change_vmax = float(cfg.hazard_change_vmax)

            if abs(self._sync_slop - old_slop) > 1e-9:
                self._rebuild_sync()
            else:
                # ensure queue limit mirrors current synchroniser
                self._ats_queue_limit = getattr(self._sync, "queue_size", 10) if self._sync else 10
        return cfg

    def _rebuild_sync(self) -> None:
        if self._sync is not None:
            try:
                self._sync.unregisterCallback(self._sync_cb)
            except Exception:
                pass
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._scan_sub, self._depth_scan_sub, self._depth_sub],
            queue_size=10,
            slop=float(self._sync_slop),
        )
        self._sync.registerCallback(self._sync_cb)
        self._ats_queue_limit = getattr(self._sync, "queue_size", 10)

    # ------------------------------------------------------------------
    def _sync_cb(self, scan: LaserScan, depth_scan: LaserScan, cloud: PointCloud2) -> None:
        clean_scan = self._sanitize_scan(scan)
        depth_clean = self._sanitize_scan(depth_scan)
        clean_scan = self._merge_depth_scan(clean_scan, depth_clean)
        now = rospy.Time.now()

        with self._lock:
            self._tracker.update_scan(clean_scan)
            self._avoider.update_scan(clean_scan)
            self._last_scan_time = now
            self._last_depth_scan_time = now

            hazard = self._process_depth_cloud(cloud)
            self._hazard_state = hazard
            self._last_depth_time = now
            try:
                queues = getattr(self._sync, "queues", [])
                self._ats_queue_size = int(sum(len(q) for q in queues))
            except AttributeError:
                self._ats_queue_size = self._ats_queue_limit

    def _odom_cb(self, msg: Odometry) -> None:
        with self._lock:
            self._avoider.update_odom(msg)

    # ------------------------------------------------------------------
    def _timer_cb(self, _event) -> None:
        with self._lock:
            now = rospy.Time.now()
            if self._last_scan_time is None or (now - self._last_scan_time) > rospy.Duration(self._scan_timeout):
                rospy.logwarn_throttle(1.0, "LiDAR scan timeout; stopping robot")
                zero = Twist()
                self._cmd_pub.publish(zero)
                self._last_cmd = zero
                self._last_track_count = 0
                self._diag.update()
                return

            if self._last_depth_time is None or (now - self._last_depth_time) > rospy.Duration(self._depth_timeout):
                rospy.logwarn_throttle(1.0, "Depth data timeout; relying on LiDAR only")
                self._hazard_state = HazardReport(notes=["depth_timeout"])  # degrade gracefully
                self._spd_scale_lp = 1.0
                self._update_hazard_change_rate(self._hazard_state)

            tracks = self._tracker.step(now.to_sec())
            self._last_track_count = len(tracks)
            barriers = self._convert_barriers(tracks)
            self._avoider.ingest_dynamic_barriers(barriers)
            self._avoider.update_nav_hint(None)

            cmd, debug = self._avoider.compute_cmd()
            cmd = self._apply_hazards(cmd, self._hazard_state)

            self._cmd_pub.publish(cmd)
            self._last_cmd = cmd
            if self._avoid_params.publish_debug:
                try:
                    base_debug = json.loads(self._avoider.to_json(debug))
                except json.JSONDecodeError:
                    base_debug = {
                        "state": debug.state,
                        "d_min": debug.d_min,
                        "chosen_theta": debug.chosen_theta,
                        "v_cmd": debug.v_cmd,
                        "w_cmd": debug.w_cmd,
                        "ttc_min": debug.ttc_min,
                        "stuck_flag": debug.stuck_flag,
                        "notes": debug.notes,
                    }
                payload = {
                    "avoid": base_debug,
                    "hazards": self._hazard_state.as_dict(),
                }
                self._debug_pub.publish(json.dumps(payload))

            self._diag.update()
            self._publish_markers(cmd)

    # ------------------------------------------------------------------
    def _sanitize_scan(self, msg: LaserScan) -> LaserScan:
        rng = np.asarray(msg.ranges, dtype=np.float32)
        rng[~np.isfinite(rng)] = msg.range_max
        rng = np.clip(rng, msg.range_min, msg.range_max)

        k = int(getattr(self._avoid_params, "median_window", 3))
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        self._avoid_params.median_window = k
        if k > 1:
            rng = self._median_filter_1d(rng, k)

        clean_scan = LaserScan()
        clean_scan.header = msg.header
        clean_scan.angle_min = msg.angle_min
        clean_scan.angle_max = msg.angle_max
        clean_scan.angle_increment = msg.angle_increment
        clean_scan.time_increment = msg.time_increment
        clean_scan.scan_time = msg.scan_time
        clean_scan.range_min = msg.range_min
        clean_scan.range_max = msg.range_max
        clean_scan.ranges = rng.tolist()
        clean_scan.intensities = list(msg.intensities)
        return clean_scan

    @staticmethod
    def _resample_scan_to_grid(
        src: LaserScan, angle_min_tgt: float, angle_inc_tgt: float, n_tgt: int
    ) -> np.ndarray:
        if not src.ranges or angle_inc_tgt <= 0.0 or src.angle_increment <= 0.0:
            return np.full(n_tgt, np.inf, dtype=np.float32)

        tgt_angles = angle_min_tgt + np.arange(n_tgt, dtype=np.float32) * angle_inc_tgt
        src_pos = (tgt_angles - src.angle_min) / src.angle_increment
        src_pos = np.clip(src_pos, 0.0, max(0.0, len(src.ranges) - 1.0))
        i0 = np.floor(src_pos).astype(np.int32)
        i1 = np.clip(i0 + 1, 0, len(src.ranges) - 1)
        w = (src_pos - i0).astype(np.float32)
        src_ranges = np.asarray(src.ranges, dtype=np.float32)
        vals = (1.0 - w) * src_ranges[i0] + w * src_ranges[i1]
        return vals

    @staticmethod
    def _median_filter_1d(arr: np.ndarray, k: int) -> np.ndarray:
        if arr.size == 0:
            return arr.astype(np.float32)
        if k <= 1:
            return arr.astype(np.float32)
        k = 2 * (k // 2) + 1
        pad = k // 2
        if pad <= 0:
            return arr.astype(np.float32)
        padded = np.pad(arr, (pad, pad), mode="edge")
        strides = (padded.strides[0], padded.strides[0])
        shape = (arr.size, k)
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        return np.median(windows, axis=1).astype(np.float32)

    @staticmethod
    def _merge_depth_scan(primary: LaserScan, secondary: LaserScan) -> LaserScan:
        if not secondary.ranges:
            return primary

        pri = np.asarray(primary.ranges, dtype=np.float32)
        if pri.size == 0:
            return primary

        resampled = DepthAvoidanceNode._resample_scan_to_grid(
            secondary,
            primary.angle_min,
            primary.angle_increment,
            pri.size,
        )

        resampled = np.clip(resampled, primary.range_min, primary.range_max)
        merged = np.minimum(pri, resampled)
        primary.ranges = merged.tolist()
        return primary

    @staticmethod
    def _compute_gate(value: float, resume: float, stop: float, inverse: bool = False) -> float:
        if inverse:
            if value <= stop:
                return 0.0
            if value >= resume:
                return 1.0
            return float(np.clip((value - stop) / max(1e-6, resume - stop), 0.0, 1.0))

        if value >= stop:
            return 0.0
        if value <= resume:
            return 1.0
        return float(np.clip((stop - value) / max(1e-6, stop - resume), 0.0, 1.0))

    def _apply_speed_lpf(self, hazard: HazardReport) -> None:
        hazard.speed_scale = float(np.clip(hazard.speed_scale, 0.0, 1.0))
        self._spd_scale_lp = (1.0 - self._spd_alpha) * self._spd_scale_lp + self._spd_alpha * hazard.speed_scale
        hazard.speed_scale = float(np.clip(self._spd_scale_lp, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _process_depth_cloud(self, cloud: PointCloud2) -> HazardReport:
        hazard = HazardReport()
        proc_start = rospy.get_time()
        stamp = cloud.header.stamp if cloud.header.stamp != rospy.Time() else rospy.Time.now()
        roi_points = np.empty((0, 3), dtype=np.float32)
        drop_points = np.empty((0, 3), dtype=np.float32)
        over_points = np.empty((0, 3), dtype=np.float32)

        if cloud.width * cloud.height == 0:
            hazard.notes.append("empty_cloud")
            self._apply_speed_lpf(hazard)
            self._update_marker_cache(stamp, roi_points, None, 0.0, drop_points, over_points)
            self._cloud_proc_ms = (rospy.get_time() - proc_start) * 1000.0
            self._update_hazard_change_rate(hazard)
            return hazard

        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_frame,
                cloud.header.frame_id or self._depth_frame,
                cloud.header.stamp,
                rospy.Duration(0.05),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._base_frame,
                    cloud.header.frame_id or self._depth_frame,
                    rospy.Time(0),
                    rospy.Duration(0.05),
                )
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
                rospy.logwarn_throttle(1.0, f"Depth TF unavailable: {exc}")
                hazard.notes.append("no_tf")
                self._apply_speed_lpf(hazard)
                self._update_marker_cache(stamp, roi_points, None, 0.0, drop_points, over_points)
                self._cloud_proc_ms = (rospy.get_time() - proc_start) * 1000.0
                self._update_hazard_change_rate(hazard)
                return hazard

        arr = ros_numpy.point_cloud2.pointcloud2_to_array(cloud)
        if arr.size == 0:
            hazard.notes.append("no_points")
            self._apply_speed_lpf(hazard)
            self._update_marker_cache(stamp, roi_points, None, 0.0, drop_points, over_points)
            self._cloud_proc_ms = (rospy.get_time() - proc_start) * 1000.0
            self._update_hazard_change_rate(hazard)
            return hazard

        xyz = ros_numpy.point_cloud2.get_xyz_points(arr, remove_nans=True)
        if xyz.size == 0:
            hazard.notes.append("no_points")
            self._apply_speed_lpf(hazard)
            self._update_marker_cache(stamp, roi_points, None, 0.0, drop_points, over_points)
            self._cloud_proc_ms = (rospy.get_time() - proc_start) * 1000.0
            self._update_hazard_change_rate(hazard)
            return hazard

        if self._sample_limit > 0 and xyz.shape[0] > self._sample_limit:
            idx = self._rng.choice(xyz.shape[0], self._sample_limit, replace=False)
            xyz = xyz[idx]

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        rot_m = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])[:3, :3].astype(np.float32)
        trans_v = np.array([translation.x, translation.y, translation.z], dtype=np.float32)
        xyz = (rot_m @ xyz.T).T + trans_v

        hazard.sample_count = int(xyz.shape[0])

        x_min = float(self._roi_cfg.get("x_min", 0.2))
        x_max = float(self._roi_cfg.get("x_max", 3.0))
        y_half = float(self._roi_cfg.get("y_halfwidth", 0.6))
        z_min = float(self._roi_cfg.get("z_min", -0.1))
        z_max = float(self._roi_cfg.get("z_max", 1.8))

        mask = (
            (xyz[:, 0] >= x_min)
            & (xyz[:, 0] <= x_max)
            & (np.abs(xyz[:, 1]) <= y_half)
            & (xyz[:, 2] >= z_min)
            & (xyz[:, 2] <= z_max)
        )
        roi_points = xyz[mask]
        if roi_points.shape[0] < self._min_points:
            hazard.notes.append("sparse_cloud")
            self._apply_speed_lpf(hazard)
            self._update_marker_cache(stamp, roi_points, None, 0.0, drop_points, over_points)
            self._cloud_proc_ms = (rospy.get_time() - proc_start) * 1000.0
            self._update_hazard_change_rate(hazard)
            return hazard

        plane_normal, plane_offset, inliers = self._estimate_ground_plane(roi_points)
        if plane_normal is None or inliers.size < max(3, self._min_points // 4):
            hazard.notes.append("ground_unstable")
            self._apply_speed_lpf(hazard)
            self._update_marker_cache(stamp, roi_points, None, 0.0, drop_points, over_points)
            self._cloud_proc_ms = (rospy.get_time() - proc_start) * 1000.0
            self._update_hazard_change_rate(hazard)
            return hazard

        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        align = float(np.clip(np.dot(plane_normal, up), -1.0, 1.0))
        slope_angle = math.degrees(math.acos(align))
        hazard.slope_angle_deg = slope_angle
        min_ground_align = 0.7  # reject near-vertical planes (cos(angle) < 0.7 => ~>45deg)
        if align < min_ground_align:
            hazard.notes.append("ground_reject_vertical")
            self._apply_speed_lpf(hazard)
            self._update_marker_cache(stamp, roi_points, None, 0.0, drop_points, over_points)
            self._cloud_proc_ms = (rospy.get_time() - proc_start) * 1000.0
            self._update_hazard_change_rate(hazard)
            return hazard

        slope_gate = self._compute_gate(
            slope_angle,
            self._hazard_params.slope_resume_deg,
            self._hazard_params.slope_stop_deg,
        )
        if slope_gate <= 0.0:
            hazard.slope_blocked = True
            hazard.notes.append("slope_blocked")
        elif slope_gate < 1.0:
            hazard.notes.append("slope_slow")

        hazard.speed_scale = min(hazard.speed_scale, slope_gate)

        ground_points = roi_points[inliers]
        ground_z = float(np.mean(ground_points[:, 2]))

        drop_gap_cells = int(self._depth_cfg.get("drop_gap_cells", 3))
        drop_thresh = self._hazard_params.drop_stop_m
        below_mask = roi_points[:, 2] < (ground_z - drop_thresh)
        drop_points = roi_points[below_mask]
        drop_depth = float(ground_z - np.min(drop_points[:, 2])) if drop_points.size else 0.0
        if drop_points.shape[0] >= max(1, drop_gap_cells):
            drop_gate = self._compute_gate(
                drop_depth,
                self._hazard_params.drop_resume_m,
                self._hazard_params.drop_stop_m,
            )
            hazard.drop_detected = drop_gate < 1.0
            hazard.drop_depth = drop_depth
            if drop_points.size:
                hazard.lateral_bias += float(np.clip(np.mean(drop_points[:, 1]), -1.0, 1.0))
            if drop_gate <= 0.0:
                hazard.notes.append("drop")
            else:
                hazard.notes.append("drop_warn")
            hazard.speed_scale = min(hazard.speed_scale, drop_gate)

        overhead_region = (
            (roi_points[:, 0] >= x_min)
            & (roi_points[:, 0] <= self._look_ahead)
            & (roi_points[:, 2] > ground_z + 0.05)
            & (roi_points[:, 2] < ground_z + self._hazard_params.overhead_resume_m)
        )
        over_points = roi_points[overhead_region]
        clearance = (
            float(np.min(over_points[:, 2] - ground_z)) if over_points.size else self._hazard_params.overhead_resume_m
        )
        over_gate = self._compute_gate(
            clearance,
            self._hazard_params.overhead_resume_m,
            self._hazard_params.overhead_stop_m,
            inverse=True,
        )
        if over_points.size and clearance <= self._hazard_params.overhead_resume_m:
            hazard.lateral_bias += float(np.clip(np.mean(over_points[:, 1]), -1.0, 1.0))
        if clearance <= self._hazard_params.overhead_stop_m:
            hazard.overhead_detected = True
            hazard.notes.append("overhang")
        hazard.overhead_clearance = clearance
        hazard.speed_scale = min(hazard.speed_scale, over_gate)

        self._apply_speed_lpf(hazard)

        self._update_marker_cache(stamp, roi_points, plane_normal, ground_z, drop_points, over_points)
        self._cloud_proc_ms = (rospy.get_time() - proc_start) * 1000.0
        self._update_hazard_change_rate(hazard)
        return hazard

    # ------------------------------------------------------------------
    def _estimate_ground_plane(
        self, points: np.ndarray, iterations: int = 40
    ) -> Tuple[Optional[np.ndarray], Optional[float], np.ndarray]:
        if points.shape[0] < 3:
            return None, None, np.array([], dtype=int)

        best_inliers: np.ndarray = np.array([], dtype=int)
        best_normal: Optional[np.ndarray] = None
        best_offset: Optional[float] = None
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        threshold = float(self._ransac_thresh)
        min_up_dot = 0.7  # reject near-vertical walls

        for _ in range(iterations):
            idx = self._rng.choice(points.shape[0], 3, replace=False)
            p0, p1, p2 = points[idx]
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-4:
                continue
            normal = normal / norm
            if normal[2] < 0.0:
                normal = -normal
            align = np.dot(normal, up)
            if align < min_up_dot:
                continue
            d = -np.dot(normal, p0)
            distances = np.abs(points @ normal + d)
            inliers = np.where(distances < threshold)[0]
            if inliers.size > best_inliers.size:
                best_inliers = inliers
                best_normal = normal
                best_offset = d

        if best_inliers.size == 0 or best_normal is None:
            return None, None, np.array([], dtype=int)

        inlier_pts = points[best_inliers]
        centroid = np.mean(inlier_pts, axis=0)
        demeaned = inlier_pts - centroid
        try:
            _, _, vh = np.linalg.svd(demeaned, full_matrices=False)
        except np.linalg.LinAlgError:
            return best_normal, best_offset, best_inliers
        normal = vh[2, :]
        if normal[2] < 0.0:
            normal = -normal
        normal = normal / np.linalg.norm(normal)
        if np.dot(normal, up) < min_up_dot:
            return None, None, np.array([], dtype=int)
        offset = -np.dot(normal, centroid)
        distances = np.abs(points @ normal + offset)
        refined_inliers = np.where(distances < threshold)[0]
        return normal, float(offset), refined_inliers

    def _update_marker_cache(
        self,
        stamp: rospy.Time,
        roi_points: np.ndarray,
        plane_normal: Optional[np.ndarray],
        ground_z: float,
        drop_points: np.ndarray,
        over_points: np.ndarray,
    ) -> None:
        markers = MarkerArray()

        roi_marker = Marker()
        roi_marker.header.frame_id = self._base_frame
        roi_marker.header.stamp = stamp
        roi_marker.ns = "depth_roi"
        roi_marker.id = 0
        roi_marker.type = Marker.LINE_LIST
        roi_marker.action = Marker.ADD
        roi_marker.scale.x = 0.01
        roi_marker.color.r = 0.2
        roi_marker.color.g = 0.8
        roi_marker.color.b = 1.0
        roi_marker.color.a = 0.4

        x_min = float(self._roi_cfg.get("x_min", 0.2))
        x_max = float(self._roi_cfg.get("x_max", 3.0))
        y_half = float(self._roi_cfg.get("y_halfwidth", 0.6))
        z_min = float(self._roi_cfg.get("z_min", -0.1))
        z_max = float(self._roi_cfg.get("z_max", 1.8))

        corners = [
            (x_min, -y_half, z_min),
            (x_min, y_half, z_min),
            (x_max, -y_half, z_min),
            (x_max, y_half, z_min),
            (x_min, -y_half, z_max),
            (x_min, y_half, z_max),
            (x_max, -y_half, z_max),
            (x_max, y_half, z_max),
        ]
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in edges:
            roi_marker.points.append(Point(*corners[i]))
            roi_marker.points.append(Point(*corners[j]))
        markers.markers.append(roi_marker)

        if plane_normal is not None:
            plane_marker = Marker()
            plane_marker.header.frame_id = self._base_frame
            plane_marker.header.stamp = stamp
            plane_marker.ns = "depth_roi"
            plane_marker.id = 1
            plane_marker.type = Marker.ARROW
            plane_marker.action = Marker.ADD
            plane_marker.scale.x = 0.04
            plane_marker.scale.y = 0.08
            plane_marker.scale.z = 0.08
            plane_marker.color.r = 0.1
            plane_marker.color.g = 1.0
            plane_marker.color.b = 0.1
            plane_marker.color.a = 0.8
            start = Point(0.0, 0.0, ground_z)
            end = Point(
                plane_normal[0] * 0.5,
                plane_normal[1] * 0.5,
                ground_z + plane_normal[2] * 0.5,
            )
            plane_marker.points = [start, end]
            markers.markers.append(plane_marker)

        if drop_points.size:
            drop_marker = Marker()
            drop_marker.header.frame_id = self._base_frame
            drop_marker.header.stamp = stamp
            drop_marker.ns = "depth_roi"
            drop_marker.id = 2
            drop_marker.type = Marker.POINTS
            drop_marker.action = Marker.ADD
            drop_marker.scale.x = 0.05
            drop_marker.scale.y = 0.05
            drop_marker.color.r = 1.0
            drop_marker.color.g = 0.1
            drop_marker.color.b = 0.1
            drop_marker.color.a = 0.9
            stride = max(1, drop_points.shape[0] // 100)
            for pt in drop_points[::stride]:
                drop_marker.points.append(Point(pt[0], pt[1], pt[2]))
            markers.markers.append(drop_marker)

        if over_points.size:
            over_marker = Marker()
            over_marker.header.frame_id = self._base_frame
            over_marker.header.stamp = stamp
            over_marker.ns = "depth_roi"
            over_marker.id = 3
            over_marker.type = Marker.POINTS
            over_marker.action = Marker.ADD
            over_marker.scale.x = 0.05
            over_marker.scale.y = 0.05
            over_marker.color.r = 1.0
            over_marker.color.g = 1.0
            over_marker.color.b = 0.1
            over_marker.color.a = 0.9
            stride = max(1, over_points.shape[0] // 100)
            for pt in over_points[::stride]:
                over_marker.points.append(Point(pt[0], pt[1], pt[2]))
            markers.markers.append(over_marker)

        self._depth_marker_cache = markers

    def _publish_markers(self, cmd: Twist) -> None:
        if self._marker_pub.get_num_connections() == 0:
            return

        markers = MarkerArray()
        if self._depth_marker_cache is not None:
            markers.markers.extend(self._depth_marker_cache.markers)

        heading = Marker()
        heading.header.frame_id = self._base_frame
        heading.header.stamp = rospy.Time.now()
        heading.ns = "depth_heading"
        heading.id = 100
        heading.type = Marker.ARROW
        heading.action = Marker.ADD
        heading.scale.x = 0.05
        heading.scale.y = 0.1
        heading.scale.z = 0.1
        heading.color.r = 0.0
        heading.color.g = 1.0
        heading.color.b = 0.0
        heading.color.a = 0.9
        heading.points = [Point(0.0, 0.0, 0.0), Point(cmd.linear.x, cmd.linear.y, 0.0)]
        markers.markers.append(heading)

        bias = Marker()
        bias.header = heading.header
        bias.ns = "depth_heading"
        bias.id = 101
        bias.type = Marker.ARROW
        bias.action = Marker.ADD
        bias.scale.x = 0.05
        bias.scale.y = 0.1
        bias.scale.z = 0.1
        bias.color.r = 1.0
        bias.color.g = 0.0
        bias.color.b = 1.0
        bias.color.a = 0.9
        bias.points = [Point(0.0, 0.0, 0.0), Point(0.0, self._hazard_state.lateral_bias, 0.0)]
        markers.markers.append(bias)

        self._marker_pub.publish(markers)

    def _update_hazard_change_rate(self, hazard: HazardReport) -> None:
        signature = (
            round(hazard.speed_scale, 2),
            hazard.slope_blocked,
            hazard.drop_detected,
            hazard.overhead_detected,
            tuple(sorted(hazard.notes)),
        )
        now = rospy.get_time()
        if self._hazard_signature != signature:
            self._hazard_signature = signature
            self._hazard_change_times.append(now)

        window = 5.0
        while self._hazard_change_times and now - self._hazard_change_times[0] > window:
            self._hazard_change_times.popleft()

        self._hazard_change_rate = (
            len(self._hazard_change_times) / window if window > 0.0 else 0.0
        )

    def _start_backoff(self, now: rospy.Time, hazard: HazardReport) -> None:
        duration = rospy.Duration(1.0)
        self._backoff_until = now + duration
        self._backoff_speed = max(0.08, min(0.18, self._avoid_params.v_max * 0.45))
        if hazard.lateral_bias:
            self._backoff_dir = float(-np.sign(hazard.lateral_bias))
        else:
            self._backoff_dir = float(self._rng.choice([-1.0, 1.0]))

    def _start_backoff(self, now: rospy.Time, hazard: HazardReport) -> None:
        duration = rospy.Duration(1.0)
        self._backoff_until = now + duration
        self._backoff_speed = max(0.08, min(0.18, self._avoid_params.v_max * 0.45))
        if hazard.lateral_bias:
            self._backoff_dir = np.sign(hazard.lateral_bias) * -1.0
        else:
            self._backoff_dir = float(self._rng.choice([-1.0, 1.0]))

    # ------------------------------------------------------------------
    def _apply_hazards(self, cmd: Twist, hazard: HazardReport) -> Twist:
        adjusted = Twist()
        adjusted.linear.x = cmd.linear.x
        adjusted.linear.y = cmd.linear.y
        adjusted.angular.z = cmd.angular.z

        effective_vmax = self._avoid_params.v_max
        if self._hazard_change_rate > self._hazard_change_limit:
            effective_vmax = min(effective_vmax, self._hazard_change_vmax)

        now = rospy.Time.now()
        if now < self._backoff_until:
            adjusted.linear.x = -min(self._backoff_speed, effective_vmax * 0.5)
            adjusted.linear.y = 0.0
            steer = (
                self._lat_gain_ang * np.clip(-hazard.lateral_bias, -1.0, 1.0)
                if hazard.lateral_bias
                else self._backoff_dir * min(0.6, self._avoid_params.w_max * 0.5)
            )
            adjusted.angular.z = np.clip(steer, -self._avoid_params.w_max, self._avoid_params.w_max)
            return adjusted

        if self._avoid_params.holonomic:
            target_y = float(
                np.clip(-hazard.lateral_bias * self._lat_gain_lin, -0.3, 0.3)
            )
            self._lateral_nudge_lp = 0.7 * self._lateral_nudge_lp + 0.3 * target_y
        else:
            self._lateral_nudge_lp = 0.0

        if hazard.speed_scale <= 0.0 or hazard.slope_blocked:
            drop_risk = hazard.drop_detected
            blocked = hazard.slope_blocked or ("overhang" in hazard.notes) or ("slope_blocked" in hazard.notes)
            if blocked and not drop_risk:
                self._start_backoff(now, hazard)
                adjusted.linear.x = -min(self._backoff_speed, effective_vmax * 0.5)
                adjusted.linear.y = 0.0
                steer = (
                    self._lat_gain_ang * np.clip(-hazard.lateral_bias, -1.0, 1.0)
                    if hazard.lateral_bias
                    else self._backoff_dir * min(0.6, self._avoid_params.w_max * 0.5)
                )
                adjusted.angular.z = np.clip(steer, -self._avoid_params.w_max, self._avoid_params.w_max)
                return adjusted
            adjusted.linear.x = 0.0
            adjusted.linear.y = 0.0
        else:
            adjusted.linear.x = float(
                np.clip(adjusted.linear.x, -effective_vmax, effective_vmax)
            )
            adjusted.linear.x *= hazard.speed_scale
            if self._avoid_params.holonomic:
                lateral_cmd = np.clip(adjusted.linear.y + self._lateral_nudge_lp, -0.3, 0.3)
                adjusted.linear.y = lateral_cmd * hazard.speed_scale
            else:
                adjusted.linear.y = 0.0

        if hazard.lateral_bias and abs(adjusted.angular.z) < self._avoid_params.w_max:
            steer = self._lat_gain_ang * np.clip(-hazard.lateral_bias, -1.0, 1.0)
            adjusted.angular.z = np.clip(
                adjusted.angular.z + steer,
                -self._avoid_params.w_max,
                self._avoid_params.w_max,
            )

        return adjusted

    # ------------------------------------------------------------------
    def _convert_barriers(self, tracks) -> List[dict]:
        barriers: List[dict] = []
        robot_speed = self._avoider.current_speed()
        for obj in tracks:
            if not obj.is_dynamic:
                continue
            radial_dir = (math.cos(obj.theta), math.sin(obj.theta))
            v_rel = obj.vx * radial_dir[0] + obj.vy * radial_dir[1] - robot_speed
            radius = max(
                self._avoid_params.proxemics_min,
                self._avoid_params.dyn_inflation_base + self._avoid_params.dyn_inflation_gain * obj.speed,
            )
            barriers.append(
                {
                    "theta": obj.theta,
                    "range": obj.range,
                    "v_rel": v_rel,
                    "radius": radius,
                }
            )
        return barriers

    # ------------------------------------------------------------------
    def _diag_inputs(self, stat):
        now = rospy.Time.now()
        scan_age = float("inf") if self._last_scan_time is None else (now - self._last_scan_time).to_sec()
        depth_age = float("inf") if self._last_depth_time is None else (now - self._last_depth_time).to_sec()
        depth_scan_age = (
            float("inf")
            if self._last_depth_scan_time is None
            else (now - self._last_depth_scan_time).to_sec()
        )

        if scan_age < self._scan_timeout and depth_age < self._depth_timeout:
            stat.summary(0, "OK")
        elif scan_age >= self._scan_timeout:
            stat.summary(1, "LiDAR timeout")
        else:
            stat.summary(1, "Depth timeout")

        stat.add("scan_age_sec", scan_age)
        stat.add("depth_age_sec", depth_age)
        stat.add("depth_scan_age_sec", depth_scan_age)
        stat.add("slope_deg", self._hazard_state.slope_angle_deg)
        stat.add("hazard_speed_scale", self._hazard_state.speed_scale)
        stat.add("hazard_notes", ",".join(self._hazard_state.notes))
        stat.add("tracked_objects", self._last_track_count)
        stat.add("hazard_sample_count", self._hazard_state.sample_count)
        stat.add("cloud_proc_ms", self._cloud_proc_ms)
        stat.add("speed_scale_lp", self._spd_scale_lp)
        stat.add("hazard_change_rate_hz", self._hazard_change_rate)
        stat.add("hazard_change_limit_hz", self._hazard_change_limit)
        stat.add("hazard_flap_vmax", self._hazard_change_vmax)
        stat.add("ats_queue_limit", self._ats_queue_limit)
        stat.add("ats_queue_len", self._ats_queue_size)
        stat.add("ats_slop", self._sync_slop)
        stat.add("cmd_linear_x", self._last_cmd.linear.x)
        stat.add("cmd_linear_y", self._last_cmd.linear.y)
        stat.add("cmd_angular_z", self._last_cmd.angular.z)
        return stat


def main() -> None:
    node = DepthAvoidanceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
