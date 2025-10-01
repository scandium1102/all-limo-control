#!/usr/bin/env python3
"""Depth-enhanced reactive avoidance node for the LIMO platform."""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
from diagnostic_updater import FunctionDiagnosticTask, Updater
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
from tf.transformations import quaternion_matrix

import rospkg
import message_filters
from dynamic_reconfigure.server import Server as DynServer
from limo_control.cfg import AvoidanceConfig

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

        cmd_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self._cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
        self._debug_pub = rospy.Publisher("~avoidance_debug_3d", String, queue_size=10)

        scan_topic = rospy.get_param("~scan_topic", "/scan")
        depth_topic = rospy.get_param("~depth_cloud_topic", "/camera/depth/points_filtered")
        depth_scan_topic = rospy.get_param("~depth_scan_topic", "/camera/depth/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")

        self._scan_sub = message_filters.Subscriber(scan_topic, LaserScan)
        self._depth_scan_sub = message_filters.Subscriber(depth_scan_topic, LaserScan)
        self._depth_sub = message_filters.Subscriber(depth_topic, PointCloud2)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._scan_sub, self._depth_scan_sub, self._depth_sub],
            queue_size=10,
            slop=self._sync_slop,
        )
        self._sync.registerCallback(self._sync_cb)

        self._odom_sub = rospy.Subscriber(odom_topic, Odometry, self._odom_cb, queue_size=10)

        self._last_scan_time: Optional[rospy.Time] = None
        self._last_depth_time: Optional[rospy.Time] = None
        self._last_depth_scan_time: Optional[rospy.Time] = None
        self._hazard_state = HazardReport()
        self._last_cmd = Twist()
        self._last_track_count: int = 0
        self._depth_scan_topic = depth_scan_topic

        self._diag = Updater()
        self._diag.setHardwareID("depth_avoidance")
        self._diag.add(FunctionDiagnosticTask("inputs", self._diag_inputs))

        self._timer = rospy.Timer(rospy.Duration(0.1), self._timer_cb)

        self._dyn_srv = DynServer(AvoidanceConfig, self._on_dyn_cfg)

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
            pad = k // 2
            if pad > 0:
                padv = np.pad(rng, (pad, pad), mode="edge")
                rng = np.array(
                    [np.median(padv[i - pad : i + pad + 1]) for i in range(pad, len(padv) - pad)],
                    dtype=np.float32,
                )

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
    def _merge_depth_scan(primary: LaserScan, secondary: LaserScan) -> LaserScan:
        if not secondary.ranges:
            return primary

        if len(primary.ranges) != len(secondary.ranges):
            return primary

        angle_tol = 1e-4
        if (
            abs(primary.angle_min - secondary.angle_min) > angle_tol
            or abs(primary.angle_increment - secondary.angle_increment) > angle_tol
        ):
            return primary

        pri = np.asarray(primary.ranges, dtype=np.float32)
        sec = np.asarray(secondary.ranges, dtype=np.float32)
        merged = np.minimum(pri, sec)
        primary.ranges = merged.tolist()
        return primary

    # ------------------------------------------------------------------
    def _process_depth_cloud(self, cloud: PointCloud2) -> HazardReport:
        hazard = HazardReport()
        if cloud.width * cloud.height == 0:
            hazard.notes.append("empty_cloud")
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
                return hazard

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        rot_m = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])[:3, :3]
        trans_v = np.array([translation.x, translation.y, translation.z], dtype=np.float32)

        points_list: List[Tuple[float, float, float]] = []
        for idx, pt in enumerate(point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)):
            if self._sample_limit > 0 and idx >= self._sample_limit:
                break
            points_list.append((float(pt[0]), float(pt[1]), float(pt[2])))

        if not points_list:
            hazard.notes.append("no_points")
            return hazard
        points = np.asarray(points_list, dtype=np.float32)

        points = (rot_m @ points.T).T + trans_v
        hazard.sample_count = int(points.shape[0])

        roi = self._depth_cfg.get("roi", {})
        x_min = float(roi.get("x_min", 0.2))
        x_max = float(roi.get("x_max", 3.0))
        y_half = float(roi.get("y_halfwidth", 0.6))
        z_min = float(roi.get("z_min", -0.1))
        z_max = float(roi.get("z_max", 1.8))

        mask = (
            (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max)
            & (np.abs(points[:, 1]) <= y_half)
            & (points[:, 2] >= z_min)
            & (points[:, 2] <= z_max)
        )
        roi_points = points[mask]
        if roi_points.shape[0] < self._min_points:
            hazard.notes.append("sparse_cloud")
            return hazard

        plane_normal, plane_offset, inliers = self._estimate_ground_plane(roi_points)
        if plane_normal is None or len(inliers) < self._min_points // 4:
            hazard.notes.append("ground_unstable")
            return hazard

        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        align = float(np.clip(np.dot(plane_normal, up), -1.0, 1.0))
        slope_angle = math.degrees(math.acos(align))
        hazard.slope_angle_deg = slope_angle
        slope_limit = float(self._depth_cfg.get("slope_max_deg", 12.0))
        if slope_angle > slope_limit:
            hazard.slope_blocked = True
            hazard.speed_scale = 0.0
            hazard.notes.append("slope_blocked")
        elif slope_angle > 0.7 * slope_limit:
            hazard.speed_scale = min(hazard.speed_scale, 0.3)
            hazard.notes.append("slope_slow")

        ground_points = roi_points[inliers]
        ground_z = float(np.mean(ground_points[:, 2]))

        drop_thresh = float(self._depth_cfg.get("drop_height_min", 0.06))
        drop_gap_cells = int(self._depth_cfg.get("drop_gap_cells", 3))
        below_mask = roi_points[:, 2] < (ground_z - drop_thresh)
        if np.count_nonzero(below_mask) >= drop_gap_cells:
            hazard.drop_detected = True
            hazard.speed_scale = 0.0
            hazard.drop_depth = float((ground_z - np.min(roi_points[below_mask, 2])))
            hazard.lateral_bias += float(np.clip(np.mean(roi_points[below_mask, 1]), -1.0, 1.0))
            hazard.notes.append("drop")

        look_ahead = float(self._depth_cfg.get("look_ahead", 1.2))
        clearance = float(self._depth_cfg.get("overhead_clearance", 0.3))
        overhead_mask = (
            (roi_points[:, 0] >= x_min)
            & (roi_points[:, 0] <= look_ahead)
            & (roi_points[:, 2] > ground_z + 0.05)
            & (roi_points[:, 2] < ground_z + clearance)
        )
        if np.count_nonzero(overhead_mask) >= max(1, drop_gap_cells):
            hazard.overhead_detected = True
            hazard.speed_scale = min(hazard.speed_scale, 0.2)
            hazard.overhead_clearance = float(np.min(roi_points[overhead_mask, 2] - ground_z))
            hazard.lateral_bias += float(np.clip(np.mean(roi_points[overhead_mask, 1]), -1.0, 1.0))
            hazard.notes.append("overhang")

        return hazard

    # ------------------------------------------------------------------
    def _estimate_ground_plane(
        self, points: np.ndarray, iterations: int = 40, threshold: float = 0.02
    ) -> Tuple[Optional[np.ndarray], Optional[float], np.ndarray]:
        best_inliers: np.ndarray = np.array([], dtype=int)
        best_normal: Optional[np.ndarray] = None
        best_offset: Optional[float] = None
        if points.shape[0] < 3:
            return None, None, best_inliers

        for _ in range(iterations):
            idx = np.random.choice(points.shape[0], 3, replace=False)
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
            d = -np.dot(normal, p0)
            distances = np.abs(points @ normal + d)
            inliers = np.where(distances < threshold)[0]
            if inliers.size > best_inliers.size:
                best_inliers = inliers
                best_normal = normal
                best_offset = d
        return best_normal, best_offset, best_inliers

    # ------------------------------------------------------------------
    def _apply_hazards(self, cmd: Twist, hazard: HazardReport) -> Twist:
        adjusted = Twist()
        adjusted.linear.x = cmd.linear.x
        adjusted.linear.y = cmd.linear.y
        adjusted.angular.z = cmd.angular.z

        if hazard.speed_scale <= 0.0:
            adjusted.linear.x = 0.0
            adjusted.linear.y = 0.0 if not self._avoid_params.holonomic else adjusted.linear.y * 0.0
        else:
            adjusted.linear.x *= hazard.speed_scale
            if self._avoid_params.holonomic:
                adjusted.linear.y *= hazard.speed_scale
            else:
                adjusted.linear.y = 0.0

        if hazard.lateral_bias and abs(adjusted.angular.z) < self._avoid_params.w_max:
            steer = 0.4 * np.clip(-hazard.lateral_bias, -1.0, 1.0)
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
        stat.add("cmd_linear_x", self._last_cmd.linear.x)
        stat.add("cmd_angular_z", self._last_cmd.angular.z)
        return stat


def main() -> None:
    node = DepthAvoidanceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
