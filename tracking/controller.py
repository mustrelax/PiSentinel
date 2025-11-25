"""
TrackingController v3: PRIMARY LOCK MODE
Key changes:
1. Simplified - no swarm centroid, just lock on ONE target
2. Ignore other drones for servo control
3. Visual tracking of all drones but servo follows only locked target
4. More stable servo control
"""
import math
import cv2
import numpy as np
from ultralytics import YOLO
from simple_pid import PID

import config
from tracking.multi_tracker import MultiObjectTracker
from tracking.threat_analyzer import SwarmAwareThreatAnalyzer


class TrackingController:
    """
    PRIMARY LOCK tracking controller.
    Locks onto one drone and ignores others for servo control.
    """
    
    def __init__(self):
        # Load YOLO model
        try:
            self.model = YOLO(config.MODEL_PATH)
            print(f"Model '{config.MODEL_PATH}' loaded successfully.")
        except Exception as e:
            print(f"FATAL ERROR: Error loading model: {e}")
            raise
        
        # Multi-object tracker
        self.mot = MultiObjectTracker()
        
        # PID controllers - with anti-windup
        self.pid_pan = PID(**config.PID_PAN, setpoint=config.FRAME_WIDTH // 2)
        self.pid_pan.output_limits = (-2.0, 2.0)
        self.pid_pan.sample_time = 0.033  # ~30fps
        
        self.pid_tilt = PID(**config.PID_TILT, setpoint=config.FRAME_HEIGHT // 2)
        self.pid_tilt.output_limits = (-2.0, 2.0)
        self.pid_tilt.sample_time = 0.033
        
        # Servo state
        self.pan_angle = float(config.PAN_START)
        self.tilt_angle = float(config.TILT_START)
        
        # Smoothing
        self.prev_pan_corr = 0.0
        self.prev_tilt_corr = 0.0
        self.MAX_DELTA_PAN = 1.5
        self.MAX_DELTA_TILT = 1.5
        
        # Target position smoothing
        self.smoothed_target_x = config.FRAME_WIDTH / 2
        self.smoothed_target_y = config.FRAME_HEIGHT / 2
        self.target_smooth_alpha = 0.4
        
        # IDLE DETECTION - stop servo when target is stable
        self.consecutive_in_deadzone = 0
        self.IDLE_THRESHOLD = 10  # frames in deadzone before full stop  # How fast to follow target
        
        # Lock state
        self.locked = False
        self.lock_start_frame = 0
        self.frame_count = 0
    
    def _apply_servo_smoothing(self, pan_corr, tilt_corr, error_radius):
        """Smooth servo corrections - MORE smooth when near target"""
        
        # ADAPTIVE ALPHA: closer to target = smoother movement
        if error_radius < 30:
            alpha = 0.1  # Very smooth when close
        elif error_radius < 60:
            alpha = 0.2
        elif error_radius < 100:
            alpha = 0.3
        else:
            alpha = 0.4  # Faster response when far
        
        smoothed_pan = alpha * pan_corr + (1 - alpha) * self.prev_pan_corr
        smoothed_tilt = alpha * tilt_corr + (1 - alpha) * self.prev_tilt_corr
        
        # Rate limiting - tighter when close to target
        if error_radius < 50:
            max_delta = 0.8  # Very slow when close
        else:
            max_delta = self.MAX_DELTA_PAN
        
        smoothed_pan = np.clip(smoothed_pan, -max_delta, max_delta)
        smoothed_tilt = np.clip(smoothed_tilt, -max_delta, max_delta)
        
        self.prev_pan_corr = smoothed_pan
        self.prev_tilt_corr = smoothed_tilt
        
        return smoothed_pan, smoothed_tilt
    
    def _get_dead_zone(self, track):
        """Get dead zone size based on track state - LARGER for stability"""
        base = 40  # Increased from 30
        
        if track is None:
            return base
        
        # Close range = larger dead zone (more tolerance)
        if SwarmAwareThreatAnalyzer.is_close_range(track):
            base = 60
        
        if track.is_aggressive_motion():
            return base + 30
        elif track.is_fast_moving():
            return base + 15
        
        return base
    
    def _smooth_target_position(self, raw_x, raw_y):
        """Apply low-pass filter to target position"""
        self.smoothed_target_x = (self.target_smooth_alpha * raw_x + 
                                  (1 - self.target_smooth_alpha) * self.smoothed_target_x)
        self.smoothed_target_y = (self.target_smooth_alpha * raw_y + 
                                  (1 - self.target_smooth_alpha) * self.smoothed_target_y)
        
        return self.smoothed_target_x, self.smoothed_target_y
    
    def _visualize_tracks(self, frame, tracks, active_track):
        """Draw all tracks - locked target in RED, others accordingly"""
        active_id = active_track.track_id if active_track else None
        
        for track in tracks:
            # Show CONFIRMED and LOST tracks
            if track.state not in [track.CONFIRMED, track.LOST]:
                continue
            
            is_locked = (track.track_id == active_id)
            is_lost = (track.state == track.LOST)
            
            box = track.get_predicted_box()
            x1, y1, x2, y2 = box
            
            if is_locked:
                if is_lost:
                    # LOCKED but LOST - Orange, tracking on prediction
                    color = (0, 165, 255)  # Orange
                    thickness = 3
                    label = f"PRED #{track.track_id}"
                else:
                    # LOCKED and CONFIRMED - Red
                    color = (0, 0, 255)
                    thickness = 3
                    label = f"LOCK #{track.track_id}"
            else:
                if is_lost:
                    # Other LOST track - Gray dashed
                    color = (128, 128, 128)
                    thickness = 1
                    label = f"LOST #{track.track_id}"
                else:
                    # Other CONFIRMED - Green
                    color = (0, 180, 0)
                    thickness = 1
                    label = f"IGN #{track.track_id}"
            
            # Draw box (dashed for LOST)
            if is_lost and not is_locked:
                # Dashed rectangle for lost tracks
                self._draw_dashed_rect(frame, (x1, y1), (x2, y2), color, thickness)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            cv2.putText(frame, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            
            # Trajectory for locked target only
            if is_locked and len(track.trajectory) > 1:
                points = [(int(x), int(y)) for x, y in track.trajectory]
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i+1], color, 2)
        
        return frame
    
    def _draw_dashed_rect(self, frame, pt1, pt2, color, thickness):
        """Draw a dashed rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        dash_length = 10
        gap_length = 5
        
        # Top edge
        for x in range(x1, x2, dash_length + gap_length):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        # Bottom edge
        for x in range(x1, x2, dash_length + gap_length):
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        # Left edge
        for y in range(y1, y2, dash_length + gap_length):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        # Right edge
        for y in range(y1, y2, dash_length + gap_length):
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def _draw_hud(self, frame, active_track, all_tracks):
        """Draw HUD with lock status"""
        panel_x = config.FRAME_WIDTH - 220
        panel_y = 10
        panel_w = 210
        panel_h = 120
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 2)
        
        # Get lock status
        lock_status = SwarmAwareThreatAnalyzer.get_lock_status(active_track, all_tracks)
        
        # Title based on state
        if active_track is None:
            title = "SEARCHING..."
            title_color = (100, 100, 100)
        elif active_track.state == active_track.LOST:
            title = "PREDICTING"
            title_color = (0, 165, 255)  # Orange
        else:
            title = "PRIMARY LOCK"
            title_color = (0, 0, 255)  # Red
        
        cv2.putText(frame, title, (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2)
        
        # Target info
        confirmed = len([t for t in all_tracks if t.state == t.CONFIRMED])
        lost = len([t for t in all_tracks if t.state == t.LOST])
        cv2.putText(frame, f"Detected: {confirmed} | Lost: {lost}", 
                   (panel_x + 10, panel_y + 48),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        if active_track:
            # Target ID and state
            state_str = "PRED" if active_track.state == active_track.LOST else "LOCK"
            cv2.putText(frame, f"Target: #{active_track.track_id} [{state_str}]", 
                       (panel_x + 10, panel_y + 68),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, title_color, 1)
            
            # Time since update (important for LOST)
            if active_track.state == active_track.LOST:
                cv2.putText(frame, f"Predicting: {active_track.time_since_update}f", 
                           (panel_x + 10, panel_y + 88),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            
            # Ignored count
            if lock_status['ignored_count'] > 0:
                cv2.putText(frame, f"Ignoring: {lock_status['ignored_count']}", 
                           (panel_x + 10, panel_y + 105),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)
            
            # Close range warning
            if lock_status['is_close_range']:
                cv2.putText(frame, "CLOSE", 
                           (panel_x + 150, panel_y + 68),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    
    def _control_servo(self, frame, active_track):
        """Servo control - follows ONLY the locked target"""
        center_x = config.FRAME_WIDTH // 2
        center_y = config.FRAME_HEIGHT // 2
        
        dead_zone = self._get_dead_zone(active_track)
        
        # Draw center crosshair
        cv2.circle(frame, (center_x, center_y), dead_zone, (100, 100, 100), 1)
        cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (100, 100, 100), 1)
        cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (100, 100, 100), 1)
        
        # NO TARGET: Hold position, decay any movement
        if active_track is None:
            self.prev_pan_corr *= 0.5
            self.prev_tilt_corr *= 0.5
            self.pid_pan.reset()
            self.pid_tilt.reset()
            return frame
        
        # Get target position (from Kalman state)
        raw_x, raw_y = active_track.kf.x[0], active_track.kf.x[1]
        
        # Smooth target position
        sx, sy = self._smooth_target_position(raw_x, raw_y)
        sx, sy = int(sx), int(sy)
        
        # Calculate error
        dx = sx - center_x
        dy = sy - center_y
        radius = math.sqrt(dx**2 + dy**2)
        
        # LOST TARGET: Different behavior based on speed history
        if active_track.state == active_track.LOST:
            avg_speed = active_track.get_average_speed()
            
            if avg_speed < 5.0:
                # STATIONARY LOST: Don't move at all, wait for re-detection
                effective_dead_zone = 9999  # Infinite dead zone = no movement
                speed_factor = 0.0
            else:
                # MOVING LOST: Slow follow on prediction
                effective_dead_zone = dead_zone * 2
                speed_factor = 0.3
        else:
            effective_dead_zone = dead_zone
            speed_factor = 1.0
        
        if radius < effective_dead_zone:
            # Inside dead zone - FULL STOP with decay
            pan_corr, tilt_corr = 0.0, 0.0
            
            # Decay previous corrections (prevents oscillation)
            self.prev_pan_corr *= 0.5
            self.prev_tilt_corr *= 0.5
            
            # Reset PID after stabilizing
            if abs(self.prev_pan_corr) < 0.01 and abs(self.prev_tilt_corr) < 0.01:
                self.pid_pan.reset()
                self.pid_tilt.reset()
        else:
            # PID control (reduced speed for LOST targets)
            pan_corr = self.pid_pan(sx) * speed_factor
            tilt_corr = self.pid_tilt(sy) * speed_factor
        
        # Apply smoothing with error radius for adaptive behavior
        pan_corr, tilt_corr = self._apply_servo_smoothing(pan_corr, tilt_corr, radius)
        
        # Update servo angles
        self.pan_angle += pan_corr
        self.tilt_angle -= tilt_corr
        
        # Clamp to limits
        self.pan_angle = np.clip(self.pan_angle, config.PAN_MIN, config.PAN_MAX)
        self.tilt_angle = np.clip(self.tilt_angle, config.TILT_MIN, config.TILT_MAX)
        
        # Visualization
        if active_track.state == active_track.LOST:
            # Orange for prediction mode
            color = (0, 165, 255)
            cv2.line(frame, (center_x, center_y), (sx, sy), color, 1)
            cv2.circle(frame, (sx, sy), 10, color, 2)
        else:
            # Red for active lock
            color = (0, 0, 255)
            cv2.line(frame, (center_x, center_y), (sx, sy), color, 2)
            size = 25
            cv2.line(frame, (sx - size, sy), (sx + size, sy), color, 3)
            cv2.line(frame, (sx, sy - size), (sx, sy + size), color, 3)
            cv2.circle(frame, (sx, sy), 6, color, -1)
        
        # Show raw vs smoothed (debug)
        raw_x_int, raw_y_int = int(raw_x), int(raw_y)
        if abs(raw_x_int - sx) > 5 or abs(raw_y_int - sy) > 5:
            cv2.circle(frame, (raw_x_int, raw_y_int), 4, (255, 255, 0), -1)
        
        return frame
    
    def process_frame(self, frame: np.ndarray):
        """
        Main processing loop.
        Simple: detect → track → lock → follow
        """
        self.frame_count += 1
        
        # YOLO detection (very low confidence for stationary objects)
        results = self.model(frame, verbose=False, conf=0.15)
        annotated = frame.copy()
        
        detections = []
        confidences = []
        raw_det_count = 0  # DEBUG
        
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            raw_det_count = len(boxes)  # DEBUG
            detections = [box.tolist() for box in boxes]
            confidences = confs.tolist()
        
        # DEBUG: Show raw YOLO count
        cv2.putText(annotated, f"YOLO:{raw_det_count}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Update MOT
        self.mot.update(detections, confidences)
        
        # Get all tracks
        all_tracks = self.mot.get_all_tracks()
        
        # Target selection (PRIMARY LOCK)
        active_track, switched = SwarmAwareThreatAnalyzer.select_active_target(
            all_tracks, 
            self.mot.active_track_id,
            self.mot.switch_cooldown_counter
        )
        
        if switched:
            self.mot.active_track_id = active_track.track_id if active_track else None
            self.mot.switch_cooldown_counter = 0
            self.lock_start_frame = self.frame_count
            
            # Reset smoothed position to new target
            if active_track:
                self.smoothed_target_x = active_track.kf.x[0]
                self.smoothed_target_y = active_track.kf.x[1]
        else:
            self.mot.switch_cooldown_counter += 1
        
        # Visualize
        annotated = self._visualize_tracks(annotated, all_tracks, active_track)
        self._draw_hud(annotated, active_track, all_tracks)
        annotated = self._control_servo(annotated, active_track)
        
        return annotated, self.pan_angle, self.tilt_angle