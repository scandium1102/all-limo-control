#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LIMO Autonomous Patrol Script (version 6).

This script enables a LIMO robot (ROS Noetic + Gazebo simulation) to perform autonomous patrol with obstacle avoidance and coverage.

Key features:
1. Sensor fusion using LiDAR and Depth camera for obstacle detection and distance evaluation.
2. Right-hand rule wall-following using a PID controller for maintaining distance to the right wall.
3. Obstacle avoidance by backing up a safe distance then turning (with a forward arc) to circumvent obstacles.
4. Stuck detection using odometry; triggers random escape maneuvers if the robot gets stuck.
5. Coverage estimation using an odometry-projected grid; triggers return to home after covering a certain area or after a time limit.
6. Homing behavior to return to the start position and stop.
7. Adaptive linear speed that decreases as obstacles get closer (to avoid collisions).
8. All major parameters are configurable via ROS parameters or dynamic reconfigure (rqt_reconfigure) for on-line tuning.

Usage:
- Place this script in `~/limo_ws/src/limo_control/scripts/limo_patrol_v6.py` and make it executable.
- Ensure the simulation environment is running (e.g., `roslaunch limo_control limo_sim_sensors.launch` to start Gazebo and sensors).
- Run this script with ROS: `rosrun limo_control limo_patrol_v6.py` (override parameters via command-line or rosparam as needed).
- Topics:
    * Subscribes to `/limo/scan` (sensor_msgs/LaserScan) for LiDAR data.
    * Subscribes to `/camera/depth/points` (sensor_msgs/PointCloud2) for depth camera point cloud.
    * Subscribes to `/odom` (nav_msgs/Odometry) for odometry data.
    * Publishes velocity commands to `/cmd_vel` (geometry_msgs/Twist).
- The node operates at ~20 Hz. Adjust `~rate_hz` parameter if needed.
- Use ROS dynamic reconfigure (rqt_reconfigure) or set ROS parameters to adjust speeds, distances, PID gains, and other settings during runtime.

Simulation start suggestion:
1. In one terminal, launch the simulation: 
   `roslaunch limo_control limo_sim_sensors.launch`
2. In another terminal, run this patrol script: 
   `rosrun limo_control limo_patrol_v6.py _patrol_duration:=180`
   (This example runs patrol for 180 seconds before returning home; adjust parameters as needed.)
3. Optionally, open `rqt_reconfigure` to fine-tune parameters on the fly (requires dynamic_reconfigure server configuration).
"""

import rospy
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs import point_cloud2
import math
import random

try:
    from dynamic_reconfigure.server import Server
    from limo_control.cfg import AvoidParamsConfig  # dynamic reconfigure config (needs separate .cfg definition in package)
    DYNAMIC_RECONFIG = True
except ImportError:
    DYNAMIC_RECONFIG = False

class PIDController:
    """Simple PID controller for maintaining wall distance."""
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.reset()
    def reset(self):
        self.integral = 0.0
        self.prev_error = None
    def update(self, error, dt):
        # PID computation
        p = self.kp * error
        self.integral += error * dt
        i = self.ki * self.integral
        d = 0.0
        if self.prev_error is not None and dt > 0:
            derivative = (error - self.prev_error) / dt
            d = self.kd * derivative
        self.prev_error = error
        output = p + i + d
        # Clamp output
        if self.output_limit is not None:
            if output > self.output_limit:
                output = self.output_limit
            elif output < -self.output_limit:
                output = -self.output_limit
        return output

class CoverageGrid:
    """Tracks visited area in a grid to estimate coverage."""
    def __init__(self, cell_size=0.5):
        self.cell_size = cell_size
        self.visited = set()
        self.min_ix = self.max_ix = None
        self.min_iy = self.max_iy = None
    def mark_visited(self, x, y):
        ix = int(math.floor(x / self.cell_size))
        iy = int(math.floor(y / self.cell_size))
        if (ix, iy) not in self.visited:
            self.visited.add((ix, iy))
            if self.min_ix is None:
                self.min_ix = self.max_ix = ix
                self.min_iy = self.max_iy = iy
            else:
                if ix < self.min_ix: self.min_ix = ix
                if ix > self.max_ix: self.max_ix = ix
                if iy < self.min_iy: self.min_iy = iy
                if iy > self.max_iy: self.max_iy = iy
    def coverage_ratio(self):
        if not self.visited:
            return 0.0
        total_cells = (self.max_ix - self.min_ix + 1) * (self.max_iy - self.min_iy + 1)
        return float(len(self.visited)) / total_cells if total_cells > 0 else 1.0

class LimoPatrolNode:
    def __init__(self):
        rospy.init_node('limo_patrol_v6', anonymous=False)
        # Parameters
        self.scan_topic = rospy.get_param("~scan_topic", "/limo/scan")
        self.cloud_topic = rospy.get_param("~cloud_topic", "/camera/depth/points")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        # Speeds
        self.forward_speed = rospy.get_param("~forward_speed", 0.25)
        self.backward_speed = rospy.get_param("~backward_speed", 0.1)
        self.turn_linear_speed = rospy.get_param("~turn_linear_speed", 0.15)
        self.turn_angular_speed = rospy.get_param("~turn_angular_speed", 0.6)
        # Distances
        self.obstacle_distance_threshold = rospy.get_param("~obstacle_distance_threshold", 0.5)
        self.deceleration_distance = rospy.get_param("~deceleration_distance", 1.0)
        self.target_wall_distance = rospy.get_param("~target_wall_distance", 0.5)
        # Sensor fusion FOV
        self.lidar_angle_range = rospy.get_param("~lidar_angle_range", 30.0)
        self.depth_y_window = rospy.get_param("~depth_y_window", 0.2)
        self.depth_horizontal_angle = rospy.get_param("~depth_horizontal_angle", 30.0)
        # Coverage and completion
        self.coverage_cell_size = rospy.get_param("~coverage_cell_size", 0.5)
        self.coverage_ratio_goal = rospy.get_param("~coverage_ratio_goal", 1.0)
        patrol_duration_param = rospy.get_param("~patrol_duration", None)
        if patrol_duration_param is not None:
            self.max_patrol_time = float(patrol_duration_param)
        else:
            self.max_patrol_time = rospy.get_param("~max_patrol_time", 0.0)
        # Stuck detection
        self.stuck_timeout = rospy.get_param("~stuck_timeout", 3.0)
        self.stuck_distance_threshold = rospy.get_param("~stuck_distance_threshold", 0.1)
        # PID gains
        kp = rospy.get_param("~wall_follow_kp", 1.2)
        ki = rospy.get_param("~wall_follow_ki", 0.0)
        kd = rospy.get_param("~wall_follow_kd", 0.2)
        max_yaw_rate = rospy.get_param("~max_yaw_rate", self.turn_angular_speed)
        self.wall_pid = PIDController(kp, ki, kd, output_limit=max_yaw_rate)
        # State variables
        self.state = "FWD"
        self.resume_state = None
        self.avoid_dir = None
        self.avoid_stage = None
        self.avoid_start_time = None
        self.stuck_stage = None
        self.stuck_turn_duration = None
        self.stuck_turn_angle = None
        # Odom tracking
        self.start_x = None
        self.start_y = None
        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.got_odom = False
        self.last_pos_for_stuck = None
        self.last_time_for_stuck = None
        # Coverage grid
        self.coverage = CoverageGrid(cell_size=self.coverage_cell_size)
        # Time
        self.start_time = rospy.Time.now()
        # Sensor data
        self.current_scan = None
        self.current_cloud = None
        # Subscribers and publisher
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=50)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        # Dynamic reconfigure
        if DYNAMIC_RECONFIG:
            self.dyn_server = Server(AvoidParamsConfig, self.dynamic_reconfig_callback)
    def dynamic_reconfig_callback(self, config, level):
        self.forward_speed = config.forward_speed
        self.backward_speed = config.backward_speed
        self.turn_linear_speed = config.turn_linear_speed
        self.turn_angular_speed = config.turn_angular_speed
        self.obstacle_distance_threshold = config.obstacle_distance_threshold
        self.deceleration_distance = config.deceleration_distance
        self.target_wall_distance = config.target_wall_distance
        self.lidar_angle_range = config.lidar_angle_range
        self.depth_y_window = config.depth_y_window
        self.depth_horizontal_angle = config.depth_horizontal_angle
        self.coverage_ratio_goal = config.coverage_ratio_goal
        self.max_patrol_time = config.patrol_duration
        self.stuck_timeout = config.stuck_timeout
        self.stuck_distance_threshold = config.stuck_distance_threshold
        # PID
        self.wall_pid.kp = config.wall_follow_kp
        self.wall_pid.ki = config.wall_follow_ki
        self.wall_pid.kd = config.wall_follow_kd
        self.wall_pid.output_limit = config.max_yaw_rate
        return config
    def scan_callback(self, msg):
        self.current_scan = msg
    def cloud_callback(self, msg):
        self.current_cloud = msg
    def odom_callback(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        # Orientation to yaw (Euler angle around Z)
        q = msg.pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        # Compute yaw from quaternion
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        # Set current pose
        self.current_x = px
        self.current_y = py
        self.current_yaw = yaw
        if not self.got_odom:
            # Record start position on first message
            self.start_x = px
            self.start_y = py
            self.got_odom = True
            # Initialize stuck detection baseline
            self.last_pos_for_stuck = (px, py)
            self.last_time_for_stuck = rospy.Time.now()
        # Mark coverage
        self.coverage.mark_visited(px, py)
    def compute_fused_obstacle_distance(self):
        """Return the nearest obstacle distance ahead (within sensor FOV), or None if none detected."""
        min_distance = float('inf')
        # LiDAR data within ±lidar_angle_range
        if self.current_scan:
            scan = self.current_scan
            angle_range_rad = math.radians(self.lidar_angle_range)
            angle = scan.angle_min
            for dist in scan.ranges:
                if angle >= -angle_range_rad and angle <= angle_range_rad:
                    if dist > 0.0 and dist < min_distance:
                        min_distance = dist
                angle += scan.angle_increment
        # Depth camera data within vertical and horizontal windows
        if self.current_cloud:
            try:
                for point in point_cloud2.read_points(self.current_cloud, field_names=("x", "y", "z"), skip_nans=True):
                    x, y, z = point[0], point[1], point[2]
                    if z <= 0:
                        continue  # ignore points behind or at camera
                    if abs(y) > self.depth_y_window:
                        continue  # outside vertical window
                    if abs(x) > math.tan(math.radians(self.depth_horizontal_angle)) * z:
                        continue  # outside horizontal FOV window
                    dist = math.sqrt(x*x + y*y + z*z)
                    if dist < min_distance:
                        min_distance = dist
                        if min_distance < 0.1:
                            # Very close obstacle, no need to search further
                            break
            except Exception as e:
                rospy.logerr("Error processing point cloud: %s", str(e))
        if min_distance == float('inf'):
            return None
        return min_distance
    def control_loop(self):
        rospy.loginfo("Waiting for initial sensor data and odometry...")
        rate = rospy.Rate(rospy.get_param("~rate_hz", 20))
        # Wait until we have at least scan and odom data
        while not rospy.is_shutdown():
            if self.current_scan is not None and self.got_odom:
                break
            rate.sleep()
        rospy.loginfo("Starting patrol control loop.")
        # Main loop
        while not rospy.is_shutdown():
            elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
            # Check if we should initiate return (time or coverage condition)
            if self.state == "FWD":
                if self.max_patrol_time > 0 and elapsed_time >= self.max_patrol_time:
                    rospy.loginfo("Patrol duration %.1f s reached, returning to start.", elapsed_time)
                    self.state = "RETURN"
                elif self.coverage_ratio_goal < 1.0:
                    current_ratio = self.coverage.coverage_ratio()
                    if current_ratio >= self.coverage_ratio_goal:
                        rospy.loginfo("Coverage goal reached (%.1f%%), returning to start.", current_ratio*100.0)
                        self.state = "RETURN"
            # Initialize a Twist command
            cmd = Twist()
            if self.state == "FWD":
                # Forward patrol with wall-follow and obstacle avoidance
                obs_dist = self.compute_fused_obstacle_distance()
                linear_spd = self.forward_speed
                if obs_dist is not None and obs_dist < self.obstacle_distance_threshold:
                    # Obstacle too close -> start avoidance maneuver
                    avoid_to_right = False
                    right_wall_dist = None
                    if self.current_scan:
                        # Check distance of wall on right side (~-90°)
                        scan = self.current_scan
                        right_center = -90.0
                        half_width = 15.0
                        min_right = float('inf')
                        angle = scan.angle_min
                        for dist in scan.ranges:
                            ang_deg = math.degrees(angle)
                            if ang_deg >= (right_center - half_width) and ang_deg <= (right_center + half_width):
                                if dist > 0.0 and dist < min_right:
                                    min_right = dist
                            angle += scan.angle_increment
                        if min_right != float('inf'):
                            right_wall_dist = min_right
                    # Decide turn direction: turn right if right side is open, otherwise left
                    if right_wall_dist is None or right_wall_dist > self.target_wall_distance * 2.0:
                        avoid_to_right = True
                    # Set avoidance state
                    self.resume_state = "FWD"
                    self.state = "AVOID"
                    self.avoid_stage = "back"
                    self.avoid_dir = "right" if avoid_to_right else "left"
                    self.avoid_start_time = rospy.Time.now()
                    # Command immediate backward motion
                    cmd.linear.x = -self.backward_speed
                    cmd.angular.z = 0.0
                else:
                    # No immediate obstacle in path
                    if obs_dist is not None and obs_dist < self.deceleration_distance:
                        # Scale forward speed as obstacle approaches
                        if obs_dist <= self.obstacle_distance_threshold:
                            linear_spd = 0.0
                        else:
                            frac = (obs_dist - self.obstacle_distance_threshold) / (self.deceleration_distance - self.obstacle_distance_threshold)
                            frac = max(0.0, min(1.0, frac))
                            linear_spd = frac * self.forward_speed
                    # Compute wall-following steering correction
                    angular_corr = 0.0
                    if self.current_scan:
                        scan = self.current_scan
                        right_center = -90.0
                        half_width = 10.0
                        min_right = float('inf')
                        angle = scan.angle_min
                        for dist in scan.ranges:
                            ang_deg = math.degrees(angle)
                            if ang_deg >= (right_center - half_width) and ang_deg <= (right_center + half_width):
                                if dist > 0.0 and dist < min_right:
                                    min_right = dist
                            angle += scan.angle_increment
                        right_dist = None if min_right == float('inf') else min_right
                        if right_dist is None:
                            right_dist = self.target_wall_distance * 2.0  # treat no wall as very far (large error negative)
                        dist_error = self.target_wall_distance - right_dist
                        angular_corr = self.wall_pid.update(dist_error, 1.0/float(rospy.get_param('~rate_hz', 20)))
                    cmd.linear.x = linear_spd
                    cmd.angular.z = angular_corr
            elif self.state == "AVOID":
                # Obstacle avoidance sequence (back then turn)
                if self.avoid_stage == "back":
                    cmd.linear.x = -self.backward_speed
                    cmd.angular.z = 0.0
                    if (rospy.Time.now() - self.avoid_start_time) >= rospy.Duration(1.0):
                        # After 1s backup, switch to turn
                        self.avoid_stage = "turn"
                        self.avoid_start_time = rospy.Time.now()
                        cmd.linear.x = self.turn_linear_speed
                        cmd.angular.z = -self.turn_angular_speed if self.avoid_dir == "right" else self.turn_angular_speed
                elif self.avoid_stage == "turn":
                    cmd.linear.x = self.turn_linear_speed
                    cmd.angular.z = -self.turn_angular_speed if self.avoid_dir == "right" else self.turn_angular_speed
                    if (rospy.Time.now() - self.avoid_start_time) >= rospy.Duration(1.0):
                        # After ~1s turn, complete avoidance
                        if self.resume_state:
                            self.state = self.resume_state
                            self.resume_state = None
                        else:
                            self.state = "FWD"
                        self.wall_pid.reset()  # reset PID after a significant heading change
                        cmd.linear.x = 0.0
                        cmd.angular.z = 0.0
            elif self.state == "STUCK":
                # Stuck recovery maneuver (random back-and-turn)
                if self.stuck_stage is None:
                    # Plan a random large turn
                    angle_to_turn = random.uniform(math.radians(90), math.radians(180))
                    direction = random.choice(["left", "right"])
                    self.stuck_turn_angle = angle_to_turn
                    self.stuck_turn_duration = angle_to_turn / self.turn_angular_speed if self.turn_angular_speed > 0 else 0
                    self.stuck_stage = "back"
                    self.avoid_dir = direction  # reuse avoid_dir for turn direction
                    self.avoid_start_time = rospy.Time.now()
                    cmd.linear.x = -self.backward_speed
                    cmd.angular.z = 0.0
                elif self.stuck_stage == "back":
                    cmd.linear.x = -self.backward_speed
                    cmd.angular.z = 0.0
                    if (rospy.Time.now() - self.avoid_start_time) >= rospy.Duration(1.0):
                        self.stuck_stage = "turn"
                        self.avoid_start_time = rospy.Time.now()
                        cmd.linear.x = self.turn_linear_speed
                        cmd.angular.z = self.turn_angular_speed if self.avoid_dir == "left" else -self.turn_angular_speed
                elif self.stuck_stage == "turn":
                    cmd.linear.x = self.turn_linear_speed
                    cmd.angular.z = self.turn_angular_speed if self.avoid_dir == "left" else -self.turn_angular_speed
                    if self.stuck_turn_duration is None or (rospy.Time.now() - self.avoid_start_time) >= rospy.Duration(self.stuck_turn_duration):
                        rospy.loginfo("Stuck recovery maneuver done, resuming.")
                        if self.resume_state:
                            self.state = self.resume_state
                            self.resume_state = None
                        else:
                            self.state = "FWD"
                        self.stuck_stage = None
                        self.wall_pid.reset()
                        cmd.linear.x = 0.0
                        cmd.angular.z = 0.0
            elif self.state == "RETURN":
                # Return-to-home mode
                if self.start_x is None or self.start_y is None:
                    rospy.logwarn("Start position unknown, cannot return to base. Stopping.")
                    self.state = "STOP"
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                else:
                    dx = self.start_x - self.current_x
                    dy = self.start_y - self.current_y
                    dist_to_home = math.hypot(dx, dy)
                    if dist_to_home < 0.2:
                        rospy.loginfo("Reached home position, stopping.")
                        self.state = "STOP"
                        cmd.linear.x = 0.0
                        cmd.angular.z = 0.0
                    else:
                        # Compute desired heading to home
                        target_yaw = math.atan2(dy, dx)
                        ang_error = math.atan2(math.sin(target_yaw - self.current_yaw), math.cos(target_yaw - self.current_yaw))
                        # P-control for heading
                        angular_cmd = ang_error
                        # Limit angular speed to turn_angular_speed
                        if angular_cmd > self.turn_angular_speed:
                            angular_cmd = self.turn_angular_speed
                        elif angular_cmd < -self.turn_angular_speed:
                            angular_cmd = -self.turn_angular_speed
                        # If heading error is large, you can reduce linear speed
                        linear_cmd = self.forward_speed
                        if abs(ang_error) > math.radians(45):
                            linear_cmd = self.forward_speed * 0.5
                        # Check for obstacles during return
                        obs_dist = self.compute_fused_obstacle_distance()
                        if obs_dist is not None and obs_dist < self.obstacle_distance_threshold:
                            # Obstacle encountered en route home – trigger avoidance
                            avoid_to_right = False
                            right_wall_dist = None
                            if self.current_scan:
                                scan = self.current_scan
                                right_center = -90.0; half_width = 15.0
                                min_right = float('inf'); angle = scan.angle_min
                                for dist in scan.ranges:
                                    ang_deg = math.degrees(angle)
                                    if ang_deg >= (right_center - half_width) and ang_deg <= (right_center + half_width):
                                        if dist > 0.0 and dist < min_right:
                                            min_right = dist
                                    angle += scan.angle_increment
                                if min_right != float('inf'):
                                    right_wall_dist = min_right
                            if right_wall_dist is None or right_wall_dist > self.target_wall_distance * 2.0:
                                avoid_to_right = True
                            self.resume_state = "RETURN"
                            self.state = "AVOID"
                            self.avoid_stage = "back"
                            self.avoid_dir = "right" if avoid_to_right else "left"
                            self.avoid_start_time = rospy.Time.now()
                            cmd.linear.x = -self.backward_speed
                            cmd.angular.z = 0.0
                        else:
                            cmd.linear.x = linear_cmd
                            cmd.angular.z = angular_cmd
            elif self.state == "STOP":
                # Stop state – patrol finished
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            # Stuck detection in forward/return states
            if self.state in ["FWD", "RETURN"]:
                if self.last_pos_for_stuck and self.last_time_for_stuck:
                    dist_moved = math.hypot(self.current_x - self.last_pos_for_stuck[0], self.current_y - self.last_pos_for_stuck[1])
                    if dist_moved >= self.stuck_distance_threshold:
                        # Significant movement -> reset baseline
                        self.last_pos_for_stuck = (self.current_x, self.current_y)
                        self.last_time_for_stuck = rospy.Time.now()
                    else:
                        if (rospy.Time.now() - self.last_time_for_stuck) >= rospy.Duration(self.stuck_timeout):
                            rospy.logwarn("Robot appears stuck (no progress in %.1f sec). Initiating recovery.", self.stuck_timeout)
                            self.resume_state = self.state
                            self.state = "STUCK"
                            self.stuck_stage = None
                            # Reset baseline to avoid immediate retrigger
                            self.last_pos_for_stuck = (self.current_x, self.current_y)
                            self.last_time_for_stuck = rospy.Time.now()
                            # Skip sending current cmd, go to next loop iteration
                            rate.sleep()
                            continue
            # Publish the velocity command
            self.cmd_pub.publish(cmd)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = LimoPatrolNode()
        node.control_loop()
    except rospy.ROSInterruptException:
        pass

