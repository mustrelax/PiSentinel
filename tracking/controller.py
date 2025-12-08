import math
import cv2
import numpy as np
import logging
import time
from simple_pid import PID
from datetime import datetime
from ultralytics import YOLO

import config
from tracking.multi_tracker import MultiObjectTracker
from tracking.threat_analyzer import SwarmAwareThreatAnalyzer
from visualization.hud import draw_hud, draw_roi_rect, draw_servo_crosshair, draw_dashed_rect

# Setup logger
logger = logging.getLogger(__name__)

class TrackingController:
    # PREPROCESSING PARAMETERS
    ENABLE_PREPROCESSING = False 
    SHOW_PREPROCESSED = False
    CLAHE_CLIP_LIMIT = config.CLAHE_CLIP_LIMIT
    CLAHE_TILE_SIZE = (8, 8)
    SHARPEN_STRENGTH = 0.5
    SHARPEN_SIGMA = 1.0
    GAMMA = config.GAMMA_CORRECTION
    
    def __init__(self):
        # Load YOLO model
        try:
            self.model = YOLO(config.MODEL_PATH)
            logger.info(f"Model '{config.MODEL_PATH}' loaded successfully.")
        except Exception as e:
            logger.error(f"FATAL ERROR: Error loading model: {e}")
            raise
        
        # Multi-object tracker
        self.mot = MultiObjectTracker()
        
        # CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=self.CLAHE_CLIP_LIMIT, 
            tileGridSize=self.CLAHE_TILE_SIZE
        )
        logger.info(f"[PREPROCESS] CLAHE enabled: clip={self.CLAHE_CLIP_LIMIT}")
        
        # PID controllers
        self.pid_pan = PID(**config.PID_PAN, setpoint=config.FRAME_WIDTH // 2)
        self.pid_pan.output_limits = (-2.0, 2.0)
        self.pid_pan.sample_time = 0.033
        
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
        
        # IDLE DETECTION
        self.consecutive_in_deadzone = 0
        self.IDLE_THRESHOLD = 10
        
        # Lock state
        self.locked = False
        self.lock_start_frame = 0
        self.frame_count = 0
        
        # Performance monitoring
        self.last_inference_ms = 0.0
        
        # Sector Scanning State
        self.sector_index = 0
        self.scan_counter = 0
        self.current_sector_name = "WIDE"
        
        # Spatial Lock State
        self.last_locked_pos = None

        # Dynamic threat assessment
        from tracking.threat_analyzer import ReAssessmentTimer, TacticalScenario
        self.reassess_timer = ReAssessmentTimer()
        self.current_scenario = TacticalScenario.SINGLE
        self.lock_age_frames = 0

        # Sentry Mode State
        self.last_detection_time = time.time()
        self.sentry_phase = 0.0
        self.last_known_pan = config.PAN_START
        self.last_known_tilt = config.TILT_START
    
    
    def _preprocess_frame(self, frame):
        if not self.ENABLE_PREPROCESSING:
            return frame
        
        # ===== STEP 1: CLAHE on L channel (LAB color space) =====
        # Convert to LAB - L channel is luminance
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel only (preserves colors)
        l_enhanced = self.clahe.apply(l_channel)
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # ===== STEP 2: Unsharp Masking (edge sharpening) =====
        # Create blurred version
        blurred = cv2.GaussianBlur(enhanced, (0, 0), self.SHARPEN_SIGMA)
        
        # Unsharp mask: original + strength * (original - blurred)
        sharpened = cv2.addWeighted(
            enhanced, 1.0 + self.SHARPEN_STRENGTH,
            blurred, -self.SHARPEN_STRENGTH,
            0
        )
        
        # Clip to valid range (prevent overflow artifacts)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # ===== STEP 3: Gamma Correction (Brighten shadows/details) =====
        # Build lookup table for gamma
        inv_gamma = 1.0 / self.GAMMA
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        
        final = cv2.LUT(sharpened, table)
        
        return final
    
    def _apply_servo_smoothing(self, pan_corr, tilt_corr, error_radius):
        # Smooth servo corrections - MORE smooth when near target
        
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
        # Get dead zone size based on track state - LARGER for stability
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
        # Apply low-pass filter to target position
        self.smoothed_target_x = (self.target_smooth_alpha * raw_x + 
                                  (1 - self.target_smooth_alpha) * self.smoothed_target_x)
        self.smoothed_target_y = (self.target_smooth_alpha * raw_y + 
                                  (1 - self.target_smooth_alpha) * self.smoothed_target_y)
        
        return self.smoothed_target_x, self.smoothed_target_y
    
    def _visualize_tracks(self, frame, tracks, active_track):
        # Draw all tracks - locked target in RED, others accordingly
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
                draw_dashed_rect(frame, (x1, y1), (x2, y2), color, thickness)
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
    
    def _get_roi_crop(self, frame, target_x, target_y):
        # Calculate ROI crop coordinates centered on target.
        h, w = frame.shape[:2]
        roi_size = config.ROI_SIZE
        half_size = roi_size // 2
        
        # Clamp center to keep ROI within frame
        cx = np.clip(target_x, half_size, w - half_size)
        cy = np.clip(target_y, half_size, h - half_size)
        
        x1 = int(cx - half_size)
        y1 = int(cy - half_size)
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        return frame[y1:y2, x1:x2], x1, y1

    def _get_sector_crop(self, frame, sector_idx):
        # Get crop for specific sector.
        # Sectors: 0=Center, 1=TL, 2=TR, 3=BL, 4=BR
        h, w = frame.shape[:2]
        roi_size = config.ROI_SIZE
        
        # Define sectors (x1, y1)
        # Center
        if sector_idx == 0:
            x1 = (w - roi_size) // 2
            y1 = (h - roi_size) // 2
            name = "CENTER"
        # Top Left
        elif sector_idx == 1:
            x1 = 0
            y1 = 0
            name = "TOP-L"
        # Top Right
        elif sector_idx == 2:
            x1 = w - roi_size
            y1 = 0
            name = "TOP-R"
        # Bottom Left
        elif sector_idx == 3:
            x1 = 0
            y1 = h - roi_size
            name = "BOT-L"
        # Bottom Right
        else: # 4
            x1 = w - roi_size
            y1 = h - roi_size
            name = "BOT-R"
            
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        return frame[y1:y2, x1:x2], x1, y1, name

    def _translate_detections_to_global(self, detections, offset_x, offset_y):
        # Convert ROI-local detections to global coordinates
        global_dets = []
        for det in detections:
            x1, y1, x2, y2 = det
            global_dets.append([
                x1 + offset_x,
                y1 + offset_y,
                x2 + offset_x,
                y2 + offset_y
            ])
        return global_dets


    

    
    def _control_servo(self, frame, active_track):
        # Servo control - follows ONLY the locked target
        center_x = config.FRAME_WIDTH // 2
        center_y = config.FRAME_HEIGHT // 2
        
        
        dead_zone = self._get_dead_zone(active_track)
        
        # Draw center crosshair using external visualizer
        draw_servo_crosshair(frame, center_x, center_y, dead_zone)
        
        # NO TARGET: Hold position, decay any movement
        # NO TARGET: Check for Sentry Mode or Hold
        # NO TARGET: Check for Sentry Mode or Hold
        if active_track is None:
            time_since_lost = time.time() - self.last_detection_time
            
            # 1. LOCAL SEARCH (Immediate reaction)
            if time_since_lost < config.LOCAL_SEARCH_TIMEOUT:
                # Scan around LAST KNOWN position
                self.sentry_phase += config.SENTRY_SCAN_SPEED * 3.0 # Faster scan for local search
                
                # Calculate target offset (smaller angle)
                target_offset = config.LOCAL_SCAN_ANGLE * math.sin(self.sentry_phase)
                
                target_pan = self.last_known_pan + target_offset
                
                # Move servo
                diff = target_pan - self.pan_angle
                self.pan_angle += diff * 0.1
                
                # Keep tilt steady at last known
                tilt_diff = self.last_known_tilt - self.tilt_angle
                self.tilt_angle += tilt_diff * 0.1
                
                cv2.putText(frame, "MODE: LOCAL SEARCH", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 2. GLOBAL SENTRY (After timeout)
            elif config.SENTRY_MODE_ENABLED and time_since_lost > config.IDLE_TIMEOUT:
                # Global Scan around CENTER
                self.sentry_phase += config.SENTRY_SCAN_SPEED
                
                # Calculate target offset
                target_offset = config.SENTRY_SCAN_ANGLE * math.sin(self.sentry_phase)
                
                # Apply to pan angle (relative to START)
                target_pan = config.PAN_START + target_offset
                
                # Smoothly move towards target pan
                diff = target_pan - self.pan_angle
                self.pan_angle += diff * 0.05 # Slower for global
                
                # Reset tilt to center
                tilt_diff = config.TILT_START - self.tilt_angle
                self.tilt_angle += tilt_diff * 0.05
                
                cv2.putText(frame, "MODE: GLOBAL SENTRY", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            else:
                # IDLE / TRANSITION
                # Just hold position and decay
                self.prev_pan_corr *= 0.5
                self.prev_tilt_corr *= 0.5
                self.pid_pan.reset()
                self.pid_tilt.reset()
                
            return frame
        
        # Update last known position when we have a target
        self.last_known_pan = self.pan_angle
        self.last_known_tilt = self.tilt_angle
        
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
            
            # SAFETY CHECK: If track is very young (low hits) and lost, DO NOT FOLLOW PREDICTION
            # This prevents the camera from swinging wildly if a false/noisy detection is locked and immediately lost.
            if active_track.hits < 10:
                effective_dead_zone = 9999
                speed_factor = 0.0
            elif avg_speed < 5.0:
                # STATIONARY LOST: Allow very slow micro-corrections to keep prediction in frame
                effective_dead_zone = dead_zone * 2
                speed_factor = 0.1 # Was 0.0 (stopped), now 0.1 (drift correction)
            else:
                # MOVING LOST: Slow follow on prediction
                effective_dead_zone = dead_zone * 2
                speed_factor = 0.3
        else:
            effective_dead_zone = dead_zone
            speed_factor = 1.0
        
        # CONTINUOUS TRACKING LOGIC
        # We removed the "hard stop" inside dead zone to prevent "stepping".
        # Instead, we let the PID controller handle small errors naturally.
        
        if radius < effective_dead_zone:
            # Inside dead zone: Reduce PID gain effectively by scaling error?
            # Or just let it be. With small dead zone (2px), it's fine.
            # We just suppress very tiny jitters.
            if radius < 1.0:
                pan_corr, tilt_corr = 0.0, 0.0
                self.pid_pan.reset()
                self.pid_tilt.reset()
            else:
                # Soft dampening inside effective zone
                pan_corr = self.pid_pan(sx) * speed_factor * 0.5
                tilt_corr = self.pid_tilt(sy) * speed_factor * 0.5
        else:
            # Normal PID control
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
    
    def toggle_preprocessing(self):
        # Toggle preprocessing on/off for A/B testing
        self.ENABLE_PREPROCESSING = not self.ENABLE_PREPROCESSING
        status = "ON" if self.ENABLE_PREPROCESSING else "OFF"
        logger.info(f"[PREPROCESS] Preprocessing toggled: {status}")
        return self.ENABLE_PREPROCESSING
    
    def toggle_show_preprocessed(self):
        # Toggle between showing original vs preprocessed frame
        self.SHOW_PREPROCESSED = not self.SHOW_PREPROCESSED
        status = "PREPROCESSED" if self.SHOW_PREPROCESSED else "ORIGINAL"
        logger.info(f"[VIEW] Now showing: {status}")
        return self.SHOW_PREPROCESSED
    
    def set_preprocessing_params(self, clip_limit=None, sharpen_strength=None):
        # Adjust preprocessing parameters at runtime
        if clip_limit is not None:
            self.CLAHE_CLIP_LIMIT = clip_limit
            self.clahe = cv2.createCLAHE(
                clipLimit=self.CLAHE_CLIP_LIMIT,
                tileGridSize=self.CLAHE_TILE_SIZE
            )
            logger.info(f"[PREPROCESS] CLAHE clip limit: {clip_limit}")
        
        if sharpen_strength is not None:
            self.SHARPEN_STRENGTH = sharpen_strength
            logger.info(f"[PREPROCESS] Sharpen strength: {sharpen_strength}")
    
    def process_frame(self, frame: np.ndarray):
        # Main processing loop with ROI Tracking and Adaptive Inference.
        self.frame_count += 1
        
        # Get current active track ID from MOT
        active_id = self.mot.active_track_id
        active_track = None
        if active_id is not None:
            # Find the track object
            for t in self.mot.tracks:
                if t.track_id == active_id:
                    active_track = t
                    break
        
        # ===== ADAPTIVE INFERENCE (Frame Skipping) =====
        # Skip detection if we have a stable lock and just detected recently
        should_detect = True
        if config.ADAPTIVE_INFERENCE and active_track and active_track.state == active_track.CONFIRMED:
            frames_since_last = self.frame_count - active_track.last_update_frame
            if frames_since_last < config.ADAPTIVE_SKIP_FRAMES:
                should_detect = False
        
        detections = []
        confidences = []
        roi_active = False
        roi_offset_x = 0
        roi_offset_y = 0
        
        if should_detect:
            # ===== ROI / CROP LOGIC =====
            # If we have a target (even if lost/predicting), crop around it
            inference_frame = frame
            
            if config.ROI_ENABLED and active_track:
                # Use predicted position for crop center
                pred_x, pred_y = active_track.kf.x[0], active_track.kf.x[1]
                
                # Check if prediction is within frame (don't crop if totally lost)
                h, w = frame.shape[:2]
                if 0 <= pred_x <= w and 0 <= pred_y <= h:
                    inference_frame, roi_offset_x, roi_offset_y = self._get_roi_crop(frame, pred_x, pred_y)
                    roi_active = True
                    self.current_sector_name = "LOCK"
            
            # ===== SECTOR SCANNING (If not locked) =====
            elif config.ZOOM_SEARCH_ENABLED and not active_track:
                # SMART ZOOM: Only zoom if we are in GLOBAL SENTRY mode (or idle long enough)
                # Don't zoom during Local Search to avoid disorientation
                time_since_lost = time.time() - self.last_detection_time
                is_global_search = time_since_lost > config.IDLE_TIMEOUT
                
                if is_global_search:
                    self.scan_counter += 1
                    
                    # Interleaved: Even frames = Wide, Odd frames = Sector Zoom
                    if self.scan_counter % 2 != 0:
                        # Zoom Scan Mode
                        inference_frame, roi_offset_x, roi_offset_y, s_name = self._get_sector_crop(frame, self.sector_index)
                        roi_active = True
                        self.current_sector_name = s_name
                        
                        # Move to next sector for next time
                        self.sector_index = (self.sector_index + 1) % 5
                    else:
                        # Wide Mode
                        self.current_sector_name = "WIDE"
                        # roi_active remains False
                else:
                    # Local search or just lost - stay wide
                    self.current_sector_name = "WIDE"
            
            # ===== PREPROCESSING =====
            processed_frame = self._preprocess_frame(inference_frame)
            
            # ===== YOLO DETECTION =====
            # DYNAMIC THRESHOLD:
            # If ROI is active (Locked), use VERY LOW threshold (trust the tracker/ROI)
            # If Searching, use normal threshold to avoid noise
            if roi_active:
                conf_thresh = config.LOCK_CONF_THRESHOLD
            else:
                conf_thresh = config.SEARCH_CONF_THRESHOLD
            
            t_inf_start = cv2.getTickCount()
            results = self.model(processed_frame, verbose=False, conf=conf_thresh)
            t_inf_end = cv2.getTickCount()
            self.last_inference_ms = (t_inf_end - t_inf_start) * 1000.0 / cv2.getTickFrequency()
            
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                
                raw_dets = [box.tolist() for box in boxes]
                raw_confs = confs.tolist()
                
                # Translate back to global coordinates if ROI was active
                if roi_active:
                    detections = self._translate_detections_to_global(raw_dets, roi_offset_x, roi_offset_y)
                else:
                    detections = raw_dets
                
                confidences = raw_confs
        
        # Update MOT (even if no detections, to run predictions)
        self.mot.update(detections, confidences)
        
        # Update last detection time if we have any confirmed tracks
        if self.mot.get_confirmed_tracks():
            self.last_detection_time = time.time()
        
        # Get all tracks
        all_tracks = self.mot.get_all_tracks()

        # Get current active track (if any)
        current_active_track = None
        if self.mot.active_track_id is not None:
            current_active_track = next(
                (t for t in all_tracks if t.track_id == self.mot.active_track_id),
                None
            )

        # === DYNAMIC THREAT ASSESSMENT ===
        # Detect scenario
        self.current_scenario = SwarmAwareThreatAnalyzer.detect_scenario(all_tracks)

        # Dynamic re-assessment
        active_track, switched = SwarmAwareThreatAnalyzer.dynamic_reassessment(
            current_active_track,
            all_tracks,
            self.frame_count,
            self.reassess_timer,
            self.current_scenario,
            self.lock_age_frames
        )

        if switched:
            self.mot.active_track_id = active_track.track_id if active_track else None
            self.mot.switch_cooldown_counter = 0
            self.lock_start_frame = self.frame_count
            self.lock_age_frames = 0

            # Reset smoothed position to new target
            if active_track:
                self.smoothed_target_x = active_track.kf.x[0]
                self.smoothed_target_y = active_track.kf.x[1]
                # Update last locked pos immediately
                self.last_locked_pos = (active_track.kf.x[0], active_track.kf.x[1])
            else:
                self.last_locked_pos = None
        else:
            self.mot.switch_cooldown_counter += 1
            self.lock_age_frames += 1
            # Update last locked pos if we have a track
            if active_track:
                self.last_locked_pos = (active_track.kf.x[0], active_track.kf.x[1])
        
        # Visualize
        annotated = frame.copy()
        
        # Draw ROI box if active (debug/feedback)
        # Draw ROI box if active (debug/feedback)
        # if roi_active:
        #     cv2.rectangle(annotated, 
        #                  (roi_offset_x, roi_offset_y), 
        #                  (roi_offset_x + config.ROI_SIZE, roi_offset_y + config.ROI_SIZE),
        #                  (0, 255, 255), 1)
        
        annotated = self._visualize_tracks(annotated, all_tracks, active_track)
        
        draw_hud(annotated, active_track, all_tracks, roi_active=roi_active, inference_ms=self.last_inference_ms, current_sector_name=self.current_sector_name)
        annotated = self._control_servo(annotated, active_track)
        
        return annotated, self.pan_angle, self.tilt_angle