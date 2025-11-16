# tracker.py - CUSTOM TRACKER (ByteTrack bypass)
import math
import numpy as np
import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from simple_pid import PID

import config

class Tracker:
    def __init__(self):
        # --- Load model ---
        try:
            self.model = YOLO(config.MODEL_PATH)
            print(f"Model '{config.MODEL_PATH}' loaded successfully.")
        except Exception as e:
            print(f"FATAL ERROR: Error loading model: {e}")
            raise
            
        # --- PID controllers ---
        self.pid_pan = PID(**config.PID_PAN, setpoint=config.FRAME_WIDTH // 2)
        self.pid_pan.output_limits = (-3, 3)

        self.pid_tilt = PID(**config.PID_TILT, setpoint=config.FRAME_HEIGHT // 2)
        self.pid_tilt.output_limits = (-3, 3)

        # --- Kalman filter (x, y, vx, vy) ---
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.zeros(4)
        self.kf.F = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], dtype=float)
        self.kf.H = np.array([[1,0,0,0], [0,1,0,0]], dtype=float)
        self.kf.P *= 1000.0
        self.kf.R = np.array([[5,0], [0,5]], dtype=float)
        self.kf.Q = np.array([[10,0,0,0], [0,10,0,0], [0,0,50,0], [0,0,0,50]], dtype=float)

        # --- State variables ---
        self.is_tracking = False
        self.lost_counter = 0
        self.pan_angle = float(config.PAN_START)
        self.tilt_angle = float(config.TILT_START)
        
        # --- Proximity tracking parameters ---
        self.max_distance_threshold = 200  # pixels - max distance to associate detection
        self.last_box_size = 50
        
        # --- Smoothing ---
        self.prev_pan_corr = 0.0
        self.prev_tilt_corr = 0.0
        
        # --- Multi-frame validation ---
        self.detection_buffer = []  # Last 3 frames
        self.confidence_threshold = 0.25

    def _apply_servo_smoothing(self, pan_corr, tilt_corr):
        """Exponential moving average for smooth servo motion."""
        alpha = 0.3
        smoothed_pan = alpha * pan_corr + (1 - alpha) * self.prev_pan_corr
        smoothed_tilt = alpha * tilt_corr + (1 - alpha) * self.prev_tilt_corr
        
        self.prev_pan_corr = smoothed_pan
        self.prev_tilt_corr = smoothed_tilt
        
        return smoothed_pan, smoothed_tilt

    def _find_closest_detection(self, boxes, predicted_x, predicted_y):
        """Find detection closest to Kalman prediction."""
        if len(boxes) == 0:
            return None, float('inf')
        
        closest_idx = None
        min_dist = float('inf')
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            dist = math.sqrt((cx - predicted_x)**2 + (cy - predicted_y)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx, min_dist

    def _find_center_detection(self, boxes):
        """Find detection closest to screen center (initial target selection)."""
        if len(boxes) == 0:
            return None
        
        cx_screen = config.FRAME_WIDTH // 2
        cy_screen = config.FRAME_HEIGHT // 2
        
        closest_idx = None
        min_dist = float('inf')
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            dist = math.sqrt((cx - cx_screen)**2 + (cy - cy_screen)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        return closest_idx

    def _track_target(self, frame, box, confidence):
        """Core tracking logic with PID control."""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Update Kalman with detection
        adaptive_R = self.kf.R.copy()
        adaptive_R *= (1.0 - confidence)
        self.kf.update(z=np.array([cx, cy]), R=adaptive_R)
        
        # Remember box size
        self.last_box_size = max(x2 - x1, y2 - y1) // 2
        
        # Distance estimation
        pixel_w = max(1, x2 - x1)
        distance_cm = (config.KNOWN_WIDTH_CM * config.FOCAL_LENGTH) / pixel_w
        
        # Use smoothed Kalman estimate for control
        sx, sy = int(self.kf.x[0]), int(self.kf.x[1])
        dx = sx - (config.FRAME_WIDTH // 2)
        dy = sy - (config.FRAME_HEIGHT // 2)
        
        # PID control
        radius = (dx * dx + dy * dy) ** 0.5
        if radius < config.DEAD_ZONE_RADIUS:
            pan_corr, tilt_corr = 0.0, 0.0
        else:
            pan_corr = self.pid_pan(sx)
            tilt_corr = self.pid_tilt(sy)
        
        # Apply smoothing
        pan_corr, tilt_corr = self._apply_servo_smoothing(pan_corr, tilt_corr)
        
        self.pan_angle += pan_corr
        self.tilt_angle -= tilt_corr
        
        # World coordinates for display
        X_cm = (dx * distance_cm) / config.FOCAL_LENGTH
        Y_cm = -(dy * distance_cm) / config.FOCAL_LENGTH
        Z_cm = distance_cm
        
        pan_rad = math.radians(self.pan_angle - 90.0)
        tilt_rad = math.radians(self.tilt_angle - 90.0)
        x_cam, y_cam, z_cam = X_cm, Y_cm, Z_cm
        x_panned = z_cam * math.sin(pan_rad) + x_cam * math.cos(pan_rad)
        z_panned = z_cam * math.cos(pan_rad) - x_cam * math.sin(pan_rad)
        Xw = x_panned
        Yw = y_cam * math.cos(tilt_rad) - z_panned * math.sin(tilt_rad)
        Zw = y_cam * math.sin(tilt_rad) + z_panned * math.cos(tilt_rad)
        dist_th = math.sqrt(Xw**2 + Yw**2 + Zw**2)
        
        # Draw tracked box (RED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(frame, (sx, sy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"TRACKING | Conf: {confidence:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"World XYZ: {Xw:.1f}, {Yw:.1f}, {Zw:.1f} cm",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Direct: {Z_cm:.1f} cm | Theoretical: {dist_th:.1f} cm",
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def _use_prediction_mode(self, frame):
        """Use Kalman prediction when no detection available."""
        # Predict position
        sx, sy = int(self.kf.x[0]), int(self.kf.x[1])
        
        # Estimate distance from last box size
        pixel_w = max(1, self.last_box_size * 2)
        distance_cm = (config.KNOWN_WIDTH_CM * config.FOCAL_LENGTH) / pixel_w
        
        dx = sx - (config.FRAME_WIDTH // 2)
        dy = sy - (config.FRAME_HEIGHT // 2)
        
        # PID control
        radius = (dx * dx + dy * dy) ** 0.5
        if radius < config.DEAD_ZONE_RADIUS:
            pan_corr, tilt_corr = 0.0, 0.0
        else:
            pan_corr = self.pid_pan(sx)
            tilt_corr = self.pid_tilt(sy)
        
        pan_corr, tilt_corr = self._apply_servo_smoothing(pan_corr, tilt_corr)
        
        self.pan_angle += pan_corr
        self.tilt_angle -= tilt_corr
        
        # World coords
        X_cm = (dx * distance_cm) / config.FOCAL_LENGTH
        Y_cm = -(dy * distance_cm) / config.FOCAL_LENGTH
        Z_cm = distance_cm
        
        pan_rad = math.radians(self.pan_angle - 90.0)
        tilt_rad = math.radians(self.tilt_angle - 90.0)
        x_panned = Z_cm * math.sin(pan_rad) + X_cm * math.cos(pan_rad)
        z_panned = Z_cm * math.cos(pan_rad) - X_cm * math.sin(pan_rad)
        Xw = x_panned
        Yw = Y_cm * math.cos(tilt_rad) - z_panned * math.sin(tilt_rad)
        Zw = Y_cm * math.sin(tilt_rad) + z_panned * math.cos(tilt_rad)
        dist_th = math.sqrt(Xw**2 + Yw**2 + Zw**2)
        
        # Draw predicted box (ORANGE)
        x1 = sx - self.last_box_size
        y1 = sy - self.last_box_size
        x2 = sx + self.last_box_size
        y2 = sy + self.last_box_size
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
        cv2.circle(frame, (sx, sy), 5, (0, 165, 255), -1)
        cv2.putText(frame, "PREDICTING", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, f"World XYZ: {Xw:.1f}, {Yw:.1f}, {Zw:.1f} cm",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Direct: {Z_cm:.1f} cm | Theoretical: {dist_th:.1f} cm",
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def process_frame(self, frame: np.ndarray):
        """
        CUSTOM PROXIMITY-BASED TRACKER
        """
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)
        annotated = frame.copy()
        
        boxes = None
        confs = None
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
        
        # --- TRACKING LOGIC ---
        if not self.is_tracking:
            # INITIAL TARGET SELECTION: closest to screen center
            if boxes is not None and len(boxes) > 0:
                idx = self._find_center_detection(boxes)
                if idx is not None:
                    x1, y1, x2, y2 = boxes[idx]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Initialize Kalman
                    self.kf.x = np.array([cx, cy, 0.0, 0.0])
                    self.pid_pan.reset()
                    self.pid_tilt.reset()
                    self.last_box_size = max(x2 - x1, y2 - y1) // 2
                    
                    self.is_tracking = True
                    self.lost_counter = 0
                    print(f"LOCKED ON TARGET at ({cx}, {cy})")
                    
                    # Draw all detections (GREEN) except locked one
                    for i, box in enumerate(boxes):
                        if i != idx:
                            x1, y1, x2, y2 = box
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    annotated = self._track_target(annotated, boxes[idx], confs[idx])
        
        else:
            # TRACKING MODE: use Kalman prediction
            self.kf.predict()
            sx, sy = int(self.kf.x[0]), int(self.kf.x[1])
            
            detection_found = False
            
            if boxes is not None and len(boxes) > 0:
                # Find closest detection to prediction
                idx, dist = self._find_closest_detection(boxes, sx, sy)
                
                # Draw all other detections (GREEN)
                for i, box in enumerate(boxes):
                    if i != idx:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if idx is not None and dist < self.max_distance_threshold:
                    # ASSOCIATION SUCCESS: track this detection
                    detection_found = True
                    annotated = self._track_target(annotated, boxes[idx], confs[idx])
                    self.lost_counter = 0
                    
                    # Add to detection buffer for validation
                    self.detection_buffer.append(True)
                    if len(self.detection_buffer) > 3:
                        self.detection_buffer.pop(0)
            
            if not detection_found:
                # NO CLOSE DETECTION: use prediction mode
                self.detection_buffer.append(False)
                if len(self.detection_buffer) > 3:
                    self.detection_buffer.pop(0)
                
                # If we had recent detections (2/3 last frames) → keep predicting
                recent_detections = sum(self.detection_buffer[-3:]) if self.detection_buffer else 0
                
                if recent_detections >= 1:  # Very tolerant: even 1/3 is enough
                    annotated = self._use_prediction_mode(annotated)
                    self.lost_counter += 1
                else:
                    # Lost too long
                    self.lost_counter += 1
                
                # Give up after 45 frames (~1.5 seconds at 30fps)
                if self.lost_counter > 45:
                    print("TARGET LOST - Searching...")
                    self.is_tracking = False
                    self.detection_buffer.clear()
        
        return annotated, self.pan_angle, self.tilt_angle