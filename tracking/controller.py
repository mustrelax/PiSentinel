"""
TrackingController: Main interface for the tracking system.
Integrates YOLO detection, MOT, PID control, and visualization.
"""
import math
import numpy as np
from collections import deque
from ultralytics import YOLO
from simple_pid import PID

import config
from tracking.multi_tracker import MultiObjectTracker
from visualization import hud, drawing_utils


class TrackingController:
    """
    Main tracking controller. This is the public API used by tracking_thread.py
    Compatible with existing codebase interface.
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
        
        # PID controllers
        self.pid_pan = PID(**config.PID_PAN, setpoint=config.FRAME_WIDTH // 2)
        self.pid_pan.output_limits = (-3, 3)
        
        self.pid_tilt = PID(**config.PID_TILT, setpoint=config.FRAME_HEIGHT // 2)
        self.pid_tilt.output_limits = (-3, 3)
        
        # Servo state
        self.pan_angle = float(config.PAN_START)
        self.tilt_angle = float(config.TILT_START)
        
        # Smoothing with rate limiting
        self.prev_pan_corr = 0.0
        self.prev_tilt_corr = 0.0
        
        # Rate limiting constants
        self.MAX_DELTA_PAN = 2.5
        self.MAX_DELTA_TILT = 2.5
    
    def _apply_servo_smoothing(self, pan_corr, tilt_corr):
        """Apply exponential moving average with rate limiting"""
        alpha = 0.25
        
        smoothed_pan = alpha * pan_corr + (1 - alpha) * self.prev_pan_corr
        smoothed_tilt = alpha * tilt_corr + (1 - alpha) * self.prev_tilt_corr
        
        # Rate limiting
        smoothed_pan = np.clip(smoothed_pan, -self.MAX_DELTA_PAN, self.MAX_DELTA_PAN)
        smoothed_tilt = np.clip(smoothed_tilt, -self.MAX_DELTA_TILT, self.MAX_DELTA_TILT)
        
        self.prev_pan_corr = smoothed_pan
        self.prev_tilt_corr = smoothed_tilt
        
        return smoothed_pan, smoothed_tilt
    
    def _get_dynamic_dead_zone(self, track):
        """Calculate dead-zone based on target motion"""
        if track is None:
            return config.DEAD_ZONE_RADIUS
        
        base_dead_zone = 25
        
        if track.is_aggressive_motion():
            return base_dead_zone + 35  # Up to 60 pixels
        elif track.is_fast_moving():
            return base_dead_zone + 20  # Up to 45 pixels
        else:
            return base_dead_zone
    
    def _visualize_all_tracks(self, frame, tracks, active_track):
        """Draw all tracks with professional styling"""
        for track in tracks:
            is_active = track.track_id == (active_track.track_id if active_track else None)
            
            # Draw box and get color
            color = drawing_utils.draw_track_box(frame, track, is_active)
            
            # Draw trajectory
            drawing_utils.draw_trajectory(frame, track, color)
            
            # Draw velocity vector for active/fast tracks
            if is_active or track.is_fast_moving():
                drawing_utils.draw_velocity_vector(frame, track, color)
        
        # Draw HUD status panel
        hud.draw_status_panel(frame, active_track, tracks)
        
        return frame
    
    def _control_servo(self, frame, active_track):
        """PID control with professional visualization"""
        center_x = config.FRAME_WIDTH // 2
        center_y = config.FRAME_HEIGHT // 2
        
        if active_track is None:
            # Draw center reticle only
            dead_zone = config.DEAD_ZONE_RADIUS
            drawing_utils.draw_center_reticle(frame, center_x, center_y, dead_zone)
            return frame
        
        sx, sy = active_track.kf.x[0], active_track.kf.x[1]
        dx = sx - center_x
        dy = sy - center_y
        
        # Dynamic dead zone
        dead_zone = self._get_dynamic_dead_zone(active_track)
        
        # PID control
        radius = math.sqrt(dx**2 + dy**2)
        if radius < dead_zone:
            pan_corr, tilt_corr = 0.0, 0.0
        else:
            pan_corr = self.pid_pan(sx)
            tilt_corr = self.pid_tilt(sy)
        
        # Apply smoothing and rate limiting
        pan_corr, tilt_corr = self._apply_servo_smoothing(pan_corr, tilt_corr)
        
        self.pan_angle += pan_corr
        self.tilt_angle -= tilt_corr
        
        # Visualization
        drawing_utils.draw_center_reticle(frame, center_x, center_y, dead_zone)
        drawing_utils.draw_tracking_line(frame, center_x, center_y, sx, sy)
        
        return frame
    
    def process_frame(self, frame: np.ndarray):
        """
        Main processing loop. This is the public API method.
        
        Args:
            frame: Input frame (numpy array)
        
        Returns:
            (annotated_frame, pan_angle, tilt_angle)
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False, conf=0.25)
        annotated = frame.copy()
        
        detections = []
        confidences = []
        
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            detections = [box.tolist() for box in boxes]
            confidences = confs.tolist()
        
        # Update multi-object tracker
        active_track = self.mot.update(detections, confidences)
        
        # Visualize all tracks
        annotated = self._visualize_all_tracks(annotated, self.mot.get_all_tracks(), active_track)
        
        # Control servo
        annotated = self._control_servo(annotated, active_track)
        
        return annotated, self.pan_angle, self.tilt_angle