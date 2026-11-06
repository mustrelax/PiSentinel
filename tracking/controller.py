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
from tracking.track_logger import TrackLoggerManager
from visualization.hud import draw_hud, draw_roi_rect, draw_servo_crosshair, draw_dashed_rect

# Setup logger
logger = logging.getLogger(__name__)

class TrackingController:
    # PREPROCESSING PARAMETERS
    ENABLE_PREPROCESSING = True 
    SHOW_PREPROCESSED = False
    PREPROCESS_LIGHTWEIGHT = False
    CLAHE_CLIP_LIMIT = config.CLAHE_CLIP_LIMIT
    CLAHE_TILE_SIZE = (8, 8)
    SHARPEN_STRENGTH = 0.5
    SHARPEN_SIGMA = 1.0
    GAMMA = config.GAMMA_CORRECTION

    @staticmethod
    def _build_gamma_lut(gamma: float):
        # Precompute LUT once; rebuilding this every frame costs CPU.
        if gamma <= 0:
            gamma = 1.0
        inv_gamma = 1.0 / gamma
        return np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
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
        self.ENABLE_PREPROCESSING = config.SIM_ENABLE_PREPROCESSING  # False=sim, True=real camera
        logger.info(f"[PREPROCESS] CLAHE enabled: clip={self.CLAHE_CLIP_LIMIT}, active={self.ENABLE_PREPROCESSING}")
        self.gamma_lut = self._build_gamma_lut(self.GAMMA)
        
        # PID controllers
        self.pid_pan = PID(**config.PID_PAN, setpoint=config.FRAME_WIDTH // 2)
        self.pid_pan.output_limits = (-5.0, 5.0)
        self.pid_pan.sample_time = 0.033
        
        self.pid_tilt = PID(**config.PID_TILT, setpoint=config.FRAME_HEIGHT // 2)
        self.pid_tilt.output_limits = (-5.0, 5.0)
        self.pid_tilt.sample_time = 0.033
        
        # Servo state
        self.pan_angle = float(config.PAN_START)
        self.tilt_angle = float(config.TILT_START)
        
        # Rate limits (used by sentry mode)
        self.MAX_DELTA_PAN = config.SERVO_RATE_LIMIT_FAR
        self.MAX_DELTA_TILT = config.SERVO_RATE_LIMIT_FAR
        
        # Feed-forward state — smoothed velocity filters servo-induced apparent motion
        self._ff_vx = 0.0
        self._ff_vy = 0.0
        self._last_ff_time = None
        # Pixels per degree: how many pixels shift when servo rotates 1 degree
        self._px_per_deg = config.FOCAL_LENGTH * math.pi / 180.0
        self.locked = False
        self.lock_start_frame = 0
        self.frame_count = 0
        
        # Performance monitoring
        self.last_inference_ms = 0.0
        self.last_track_ms = 0.0
        self.last_draw_ms = 0.0
        self.inference_ran = False

        # Rolling health-stats. Populated each process_frame tick; consumed
        # by the periodic unitHealth emit. 30-second rolling windows for
        # FPS / detection-rate so the operator sees a stable number rather
        # than per-frame jitter.
        self._health_start_ts = time.time()
        self._health_frame_times = []     # wallclock timestamps, capped at 30s
        self._health_detect_times = []    # timestamps of frames where inference produced ≥1 detection
        self._health_last_inference_ran = False
        
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
        self.previous_scenario = TacticalScenario.SINGLE  # For change detection
        self.lock_age_frames = 0

        # Sentry Mode State
        self.last_detection_time = time.time()
        self.sentry_phase = 0.0
        self.last_known_pan = config.PAN_START
        self.last_known_tilt = config.TILT_START
        
        # Alert callback (set by main.py to notify central server)
        self.on_track_confirmed = None
        self.on_tracking_status = None  # Callback for LOCKED/ACTIVE/LOST status
        self.track_status_cache = {}    # {track_id: last_status} for state change detection
        self.on_telemetry_update = None # Callback for Azimuth/Distance
        # Raw spherical telemetry for any confirmed track (consumed by GeolocationManager).
        # Signature: (track_id, pan_deg, tilt_deg, optical_az_deg, optical_elev_deg, range_m, track_state)
        #   track_state is 'LOCKED' for the engaged primary or 'PASSIVE' for the rest.
        # The primary fires at ~6 Hz from _calculate_and_send_telemetry; non-primary
        # confirmed tracks fire at ~2 Hz from the status-reporting loop in process_frame.
        self.on_active_telemetry = None
        self._passive_telemetry_frame_counter = 0
        self.on_track_deleted = None    # (track_id) -> notify peers this track ended
        # Periodic unit telemetry (always-on, even without a tracked drone). Lets the
        # command-center map render the camera's live look bearing during sentry/scan.
        # Signature: (pan_angle, tilt_angle, has_active_track: bool)
        self.on_unit_telemetry = None
        self._unit_telemetry_frame_counter = 0

        # Periodic Pi health snapshot for the Sensor Health dashboard (1 Hz).
        # Signature: (health_dict)
        self.on_unit_health = None
        self._unit_health_frame_counter = 0
        
        self.alerted_track_ids = set()  # Prevent duplicate alerts
        
        # State tracking for status reporting
        self.last_reported_track = None 
        self.last_reported_status = None
        self.telemetry_frame_counter = 0
        
        # Track Logging (CSV + Server)
        self.track_logger_manager = TrackLoggerManager(config.DEVICE_NAME)
        self.on_track_log_complete = None  # Set by main.py
        self._last_annotated_frame = None  # For false-positive screenshots
        self.pending_switch_track_id = None
        self.pending_switch_count = 0
    
    def get_health_snapshot(self):
        """Compose a unitHealth payload for the central server dashboard.

        Returns a dict with rolling 30-second FPS, detection rate, and the
        current LOCKED engagement summary (track_id + σ on that track).
        Called periodically (1 Hz) by main.py / central_connector.
        """
        now = time.time()
        window_s = 30.0
        recent_frames = [t for t in self._health_frame_times if t > now - window_s]
        recent_dets = [t for t in self._health_detect_times if t > now - window_s]
        span = max(1.0, (recent_frames[-1] - recent_frames[0])) if len(recent_frames) >= 2 else window_s
        fps = (len(recent_frames) - 1) / span if len(recent_frames) >= 2 else 0.0
        det_per_s = len(recent_dets) / window_s

        # Active LOCKED track summary (if any).
        active_id = self.mot.active_track_id
        active_sigma = None
        if active_id is not None:
            for t in self.mot.tracks:
                if t.track_id == active_id and hasattr(t, 'real_pos'):
                    # Approximate σ as a fraction of range — matches Pi-side
                    # config k_range for consistency with WorldEstimate.
                    rng = float(t.real_pos[2]) if t.real_pos[2] > 0 else 0.0
                    active_sigma = max(config.GEOLOC_MIN_UNCERTAINTY_M, config.GEOLOC_DIST_UNCERTAINTY * rng)
                    break

        # Count confirmed PASSIVE tracks (everything confirmed except the active).
        passive_count = 0
        for t in self.mot.tracks:
            if getattr(t, 'state', None) == getattr(t, 'CONFIRMED', None) and t.track_id != active_id:
                passive_count += 1

        # CPU temperature — Raspberry Pi exposes thermal zone 0 in millidegrees.
        # Best-effort; returns None on non-Pi hosts or read failure (laptop sim,
        # permissions, etc.) so the dashboard just hides the field cleanly.
        cpu_temp_c = None
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as fh:
                cpu_temp_c = int(fh.read().strip()) / 1000.0
        except Exception:
            cpu_temp_c = None

        return {
            'uptime_s': now - self._health_start_ts,
            'fps_30s': fps,
            'detect_per_s_30s': det_per_s,
            'active_track_id': active_id,
            'active_track_sigma': active_sigma,
            'passive_track_count': passive_count,
            'total_frames': self.frame_count,
            'cpu_temp_c': cpu_temp_c,
        }

    def finalize_on_shutdown(self):
        """Finalize all active track logs on shutdown/disconnect."""
        summaries = self.track_logger_manager.finalize_all("DISCONNECTED_LOST")
        for summary in summaries:
            if self.on_track_log_complete:
                try:
                    self.on_track_log_complete(summary)
                except Exception as e:
                    logger.warning(f"Failed to send disconnect summary: {e}")
        return summaries
        
    def _check_threat(self, track, frame):
        """
        Visualizes high threat levels using SwarmAwareThreatAnalyzer.
        """
        from tracking.threat_analyzer import SwarmAwareThreatAnalyzer, ThreatZone
        
        center_x = config.FRAME_WIDTH / 2
        center_y = config.FRAME_HEIGHT / 2
        
        try:
            zone = SwarmAwareThreatAnalyzer.classify_threat_zone(track, center_x, center_y)
            
            if zone == ThreatZone.RED:
                # Flashing Red Border or Text
                cv2.putText(frame, "THREAT: RED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.rectangle(frame, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), (0, 0, 255), 4)
            elif zone == ThreatZone.YELLOW:
                cv2.putText(frame, "THREAT: YELLOW", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
        except Exception as e:
            logger.error(f"Threat check failed: {e}")

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
        if self.PREPROCESS_LIGHTWEIGHT:
            # Lightweight mode keeps contrast help but avoids heavy edge boosting noise.
            sharpened = enhanced
        else:
            blurred = cv2.GaussianBlur(enhanced, (0, 0), self.SHARPEN_SIGMA)
            sharpened = cv2.addWeighted(
                enhanced, 1.0 + self.SHARPEN_STRENGTH,
                blurred, -self.SHARPEN_STRENGTH,
                0
            )
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # ===== STEP 3: Gamma Correction (Brighten shadows/details) =====
        final = cv2.LUT(sharpened, self.gamma_lut)
        
        return final
    
    def _apply_servo_smoothing(self, pan_corr, tilt_corr, error_radius):
        # Adaptive rate limiting — gentle near center, aggressive when far
        if error_radius < config.GAIN_SCHEDULE_NEAR:
            max_delta = config.SERVO_RATE_LIMIT_CLOSE
        elif error_radius > config.GAIN_SCHEDULE_FAR:
            max_delta = config.SERVO_RATE_LIMIT_FAR
        else:
            t = (error_radius - config.GAIN_SCHEDULE_NEAR) / (config.GAIN_SCHEDULE_FAR - config.GAIN_SCHEDULE_NEAR)
            max_delta = config.SERVO_RATE_LIMIT_CLOSE + t * (config.SERVO_RATE_LIMIT_FAR - config.SERVO_RATE_LIMIT_CLOSE)

        # Track if rate limiter clipped the output (for anti-windup)
        was_clipped = (abs(pan_corr) > max_delta) or (abs(tilt_corr) > max_delta)

        pan_corr = np.clip(pan_corr, -max_delta, max_delta)
        tilt_corr = np.clip(tilt_corr, -max_delta, max_delta)

        return pan_corr, tilt_corr, was_clipped
    
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
                else: # LOCKED and CONFIRMED
                    if track.state == track.CONFIRMED:
                        # Drawing setup
                        color = (0, 0, 255) # RED (User request)
                        thickness = 2
                        
                        # --- MONOCULAR ESTIMATION ---
                        # --- MONOCULAR ESTIMATION ---
                        # Assumptions for Academic Pinhole Model:
                        # 1. Standard Drone Width used as reference  
                        # 2. Focal Length from config
                        REAL_DRONE_WIDTH = config.KNOWN_WIDTH_CM / 100.0  # Convert cm to m
                        FOCAL_LENGTH = config.FOCAL_LENGTH
                        CENTER_X = config.FRAME_WIDTH / 2
                        CENTER_Y = config.FRAME_HEIGHT / 2
                        
                        # Calculate accurate DT
                        current_time = time.time()
                        dt = current_time - self.last_detection_time
                        if dt > 0.1 or dt <= 0: # Sanity check (max lag 100ms)
                             dt = 1.0/60.0 # Default to 60fps
                        
                        track.estimate_3d_state(FOCAL_LENGTH, REAL_DRONE_WIDTH, CENTER_X, CENTER_Y, dt=dt)
                        
                        real_dist = track.real_pos[2] if hasattr(track, 'real_pos') else 0.0
                        real_speed = track.get_real_speed()
                        azimuth = track.get_azimuth()
                        
                        # Send Telemetry (Via Callbacks)
                        if self.on_tracking_status:
                            self.on_tracking_status(track.track_id, "ACTIVE")
                            
                        # Calculated Azimuth, Distance, and Speed
                        if self.on_telemetry_update:
                            self.on_telemetry_update(track.track_id, azimuth, real_dist, real_speed)
                        
                        # Label with threat score
                        score = getattr(track, 'threat_score', 0)
                        label = f"ID {track.track_id} [{score:.0f}]"
                        
                        # Threat logic (Internal state update only, visualization moved to HUD)
                        # self._check_threat(track, frame)
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
                    score = getattr(track, 'threat_score', 0)
                    label = f"#{track.track_id} [{score:.0f}]"
            
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
    
    def _calculate_and_send_telemetry(self, track):
        # Phase 1: Minimal Radar Telemetry
        self.telemetry_frame_counter += 1
        if self.telemetry_frame_counter % 5 != 0:
            return

        if not self.on_telemetry_update or not track:
            return

        # 1. Total Azimuth (Servo + Optical)
        # Pan 90 = Center. Pan < 90 is Right. 
        # Standard Bearing: Clockwise from North? 
        # Here: Relative to Robot Center (0 deg). Right is positive.
        
        # Servo Looking Right (e.g. 80) -> 90 - 80 = +10 deg
        servo_offset = 90.0 - self.pan_angle
        
        # Optical Right (Positive x) -> + deg
        optical_azimuth = track.get_azimuth()
        
        true_azimuth = servo_offset + optical_azimuth
        
        # Store for HUD (so we can see what we send)
        track.system_azimuth = true_azimuth
        
        # 2. Distance & Speed (3D Kalman Filter)
        distance = track.real_pos[2] if hasattr(track, 'real_pos') else 0.0
        speed = track.get_real_speed()

        # 3. Send
        try:
             self.on_telemetry_update(track.track_id, true_azimuth, distance, speed)
        except Exception as e:
             logger.warning(f"Telemetry callback failed: {e}")

        # 4. Forward raw spherical values to the geolocation layer.
        # Distance must be real (>0) before a world estimate is meaningful.
        if self.on_active_telemetry and distance > 0.0:
            try:
                optical_elev = track.get_elevation()
                self.on_active_telemetry(
                    track.track_id,
                    float(self.pan_angle),
                    float(self.tilt_angle),
                    float(optical_azimuth),
                    float(optical_elev),
                    float(distance),
                    "LOCKED",
                )
            except Exception as e:
                logger.warning(f"Active telemetry callback failed: {e}")


    def _control_servo(self, frame, active_track, all_tracks=None):
        # Servo control - follows ONLY the locked target
        center_x = config.FRAME_WIDTH // 2
        center_y = config.FRAME_HEIGHT // 2
        
        
        dead_zone = self._get_dead_zone(active_track)
        
        # Draw center crosshair using external visualizer
        draw_servo_crosshair(frame, center_x, center_y, dead_zone)
        
        # NO TARGET: Check for Sentry Mode or Hold
        if active_track is None:
            if all_tracks and len(all_tracks) > 0:
                cv2.putText(frame, "HOLD: ACQUIRING...", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                return frame
                
            time_since_lost = time.time() - self.last_detection_time
            
            # 1. LOCAL SEARCH (Immediate reaction)
            if time_since_lost < config.LOCAL_SEARCH_TIMEOUT:
                # Scan around LAST KNOWN position
                self.sentry_phase += config.SENTRY_SCAN_SPEED * 3.0 # Faster scan for local search
                self.sentry_phase = self.sentry_phase % (2.0 * math.pi)

                # Calculate target offset (smaller angle)
                target_offset = config.LOCAL_SCAN_ANGLE * math.sin(self.sentry_phase)

                target_pan = self.last_known_pan + target_offset

                # Move servo (clamped to rate limit)
                diff = target_pan - self.pan_angle
                step = max(-self.MAX_DELTA_PAN, min(self.MAX_DELTA_PAN, diff * 0.1))
                self.pan_angle += step

                # Keep tilt steady at last known (clamped)
                tilt_diff = self.last_known_tilt - self.tilt_angle
                tilt_step = max(-self.MAX_DELTA_TILT, min(self.MAX_DELTA_TILT, tilt_diff * 0.1))
                self.tilt_angle += tilt_step
                
                cv2.putText(frame, "MODE: LOCAL SEARCH", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 2. GLOBAL SENTRY (After timeout)
            elif config.SENTRY_MODE_ENABLED and time_since_lost > config.IDLE_TIMEOUT:
                # Global Scan around CENTER
                self.sentry_phase += config.SENTRY_SCAN_SPEED
                self.sentry_phase = self.sentry_phase % (2.0 * math.pi)

                # Calculate target offset
                target_offset = config.SENTRY_SCAN_ANGLE * math.sin(self.sentry_phase)

                # Apply to pan angle (relative to START)
                target_pan = config.PAN_START + target_offset

                # Smoothly move towards target pan (clamped to rate limit)
                diff = target_pan - self.pan_angle
                step = max(-self.MAX_DELTA_PAN, min(self.MAX_DELTA_PAN, diff * 0.05))
                self.pan_angle += step

                # Reset tilt to center (clamped)
                tilt_diff = config.TILT_START - self.tilt_angle
                tilt_step = max(-self.MAX_DELTA_TILT, min(self.MAX_DELTA_TILT, tilt_diff * 0.05))
                self.tilt_angle += tilt_step
                
                cv2.putText(frame, "MODE: GLOBAL SENTRY", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            else:
                # IDLE / TRANSITION
                self.pid_pan.reset()
                self.pid_tilt.reset()
                
            return frame
        
        # Update last known position when we have a target
        self.last_known_pan = self.pan_angle
        self.last_known_tilt = self.tilt_angle
        
        # Get target position directly from Kalman state (no EMA layer)
        sx = int(active_track.kf.x[0])
        sy = int(active_track.kf.x[1])
        
        # Get velocity from Kalman state (px/sec)
        vx = active_track.kf.x[2]
        vy = active_track.kf.x[3]
        
        # Latency compensation: predict position forward by capture delay
        # Compensates for HDMI capture card + inference pipeline latency
        if config.LATENCY_COMPENSATION_ENABLED:
            dt_comp = config.LATENCY_COMPENSATION_MS / 1000.0
            sx = int(active_track.kf.x[0] + vx * dt_comp)
            sy = int(active_track.kf.x[1] + vy * dt_comp)
        
        # Calculate error from frame center
        dx = sx - center_x
        dy = sy - center_y
        radius = math.sqrt(dx**2 + dy**2)
        
        # LOST TARGET: Different behavior based on speed history
        if active_track.state == active_track.LOST:
            avg_speed = active_track.get_average_speed()
            if active_track.hits < 10:
                effective_dead_zone = 9999
                speed_factor = 0.0
            elif avg_speed < 5.0 * config.TARGET_FPS:
                effective_dead_zone = dead_zone * 2
                speed_factor = 0.1
            else:
                effective_dead_zone = dead_zone * 2
                speed_factor = 0.3
        else:
            effective_dead_zone = dead_zone
            speed_factor = 1.0
        
        # === GAIN SCHEDULING (Eq. 29-30) ===
        # Kp adapts to error distance: gentle when centered, aggressive when far
        if config.GAIN_SCHEDULE_ENABLED:
            if radius < config.GAIN_SCHEDULE_NEAR:
                kp = config.KP_CLOSE
            elif radius > config.GAIN_SCHEDULE_FAR:
                kp = config.KP_FAR
            else:
                t = (radius - config.GAIN_SCHEDULE_NEAR) / (config.GAIN_SCHEDULE_FAR - config.GAIN_SCHEDULE_NEAR)
                kp = config.KP_CLOSE + t * (config.KP_FAR - config.KP_CLOSE)
            self.pid_pan.Kp = kp
            self.pid_tilt.Kp = kp
        
        # === SAVE INTEGRAL STATE (for anti-windup, Eq. 21) ===
        prev_integral_pan = getattr(self.pid_pan, '_integral', 0.0)
        prev_integral_tilt = getattr(self.pid_tilt, '_integral', 0.0)
        
        # === PID COMPUTATION (Eq. 12) ===
        if radius < effective_dead_zone:
            if radius < 8.0:
                # Dead stop — sub-pixel jitter from Kalman, zero it out
                pan_corr, tilt_corr = 0.0, 0.0
                self.pid_pan.reset()
                self.pid_tilt.reset()
            else:
                # Soft dampening inside dead zone
                pan_corr = self.pid_pan(sx) * speed_factor * 0.5
                tilt_corr = self.pid_tilt(sy) * speed_factor * 0.5
        else:
            # Normal PID control
            pan_corr = self.pid_pan(sx) * speed_factor
            tilt_corr = self.pid_tilt(sy) * speed_factor
        
        # === FEED-FORWARD (Eq. 25-26) ===
        # Kalman velocity (vx, vy) is in px/SEC. We need degrees/frame.
        # Problem: raw Kalman velocity includes servo-induced apparent motion.
        # Solution: heavy EMA filters out short-lived servo jitter (oscillates
        # around zero for stationary drones) while preserving sustained drone velocity.
        self._ff_vx = config.FF_SMOOTH_ALPHA * vx + (1.0 - config.FF_SMOOTH_ALPHA) * self._ff_vx
        self._ff_vy = config.FF_SMOOTH_ALPHA * vy + (1.0 - config.FF_SMOOTH_ALPHA) * self._ff_vy
        
        ff_speed = math.sqrt(self._ff_vx**2 + self._ff_vy**2)
        
        if ff_speed > config.FF_MIN_SPEED and speed_factor > 0:
            # Compute actual dt for frame-rate independent FF
            now = time.monotonic()
            if self._last_ff_time is None:
                dt_ff = 1.0 / 30.0
            else:
                dt_ff = max(0.001, min(now - self._last_ff_time, 0.2))
            self._last_ff_time = now
            
            # Convert: px/sec * sec/frame / (px/deg) = deg/frame
            ff_pan = -(self._ff_vx * dt_ff / self._px_per_deg) * config.FF_GAIN_PAN * speed_factor
            ff_tilt = -(self._ff_vy * dt_ff / self._px_per_deg) * config.FF_GAIN_TILT * speed_factor
        else:
            ff_pan = 0.0
            ff_tilt = 0.0
            # Keep time updated even when FF is off
            self._last_ff_time = time.monotonic()
        
        # Combined output: PID (reactive) + FF (proactive)
        pan_corr += ff_pan
        tilt_corr += ff_tilt
        
        # === RATE LIMITING + ANTI-WINDUP (Eq. 14, 21) ===
        pan_corr, tilt_corr, saturated = self._apply_servo_smoothing(pan_corr, tilt_corr, radius)
        
        # Anti-windup: freeze integral when rate limiter clips output
        if saturated:
            try:
                self.pid_pan._integral = prev_integral_pan
                self.pid_tilt._integral = prev_integral_tilt
            except AttributeError:
                pass
        
        # Update servo angles
        self.pan_angle += pan_corr
        self.tilt_angle -= tilt_corr
        
        # Clamp to servo limits
        self.pan_angle = np.clip(self.pan_angle, config.PAN_MIN, config.PAN_MAX)
        self.tilt_angle = np.clip(self.tilt_angle, config.TILT_MIN, config.TILT_MAX)
        
        # Visualization
        if active_track.state == active_track.LOST:
            color = (0, 165, 255)  # Orange for prediction mode
            cv2.line(frame, (center_x, center_y), (sx, sy), color, 1)
            cv2.circle(frame, (sx, sy), 10, color, 2)
        else:
            color = (0, 0, 255)  # Red for active lock
            cv2.line(frame, (center_x, center_y), (sx, sy), color, 2)
            size = 25
            cv2.line(frame, (sx - size, sy), (sx + size, sy), color, 3)
            cv2.line(frame, (sx, sy - size), (sx, sy + size), color, 3)
            cv2.circle(frame, (sx, sy), 6, color, -1)

        
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
        self.inference_ran = False

        # Rolling health-stat update: record this frame for FPS computation
        # and prune the window. Detection count is added later in the loop
        # whenever a YOLO inference run yields ≥1 detection.
        now_wall = time.time()
        self._health_frame_times.append(now_wall)
        cutoff = now_wall - 30.0
        while self._health_frame_times and self._health_frame_times[0] < cutoff:
            self._health_frame_times.pop(0)
        while self._health_detect_times and self._health_detect_times[0] < cutoff:
            self._health_detect_times.pop(0)

        # Get current active track ID from MOT
        active_id = self.mot.active_track_id
        active_track = None
        if active_id is not None:
            # Find the track object
            for t in self.mot.tracks:
                if t.track_id == active_id:
                    active_track = t
                    break

        # Always-on unit telemetry (10 Hz at 30 fps). Drives the live look-direction
        # wedge on the command-center map even when no drone is being tracked.
        self._unit_telemetry_frame_counter += 1
        if self.on_unit_telemetry and self._unit_telemetry_frame_counter % 3 == 0:
            try:
                self.on_unit_telemetry(
                    float(self.pan_angle),
                    float(self.tilt_angle),
                    active_track is not None,
                )
            except Exception as e:
                logger.warning(f"on_unit_telemetry callback failed: {e}")

        # Periodic Pi health emit (1 Hz at 30 fps). Feeds the Sensor Health
        # dashboard so the operator notices a degrading Pi (low FPS, no
        # detections, etc.) before it costs a track.
        self._unit_health_frame_counter += 1
        if self.on_unit_health and self._unit_health_frame_counter % 30 == 0:
            try:
                self.on_unit_health(self.get_health_snapshot())
            except Exception as e:
                logger.warning(f"on_unit_health callback failed: {e}")
        
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
            if roi_active:
                conf_thresh = config.LOCK_CONF_THRESHOLD
            else:
                conf_thresh = config.SEARCH_CONF_THRESHOLD
            
            t_inf_start = cv2.getTickCount()
            results = self.model(processed_frame, verbose=False, conf=conf_thresh)
            t_inf_end = cv2.getTickCount()
            self.last_inference_ms = (t_inf_end - t_inf_start) * 1000.0 / cv2.getTickFrequency()
            self.inference_ran = True
            
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
        
        # ===== TRACKING LOGIC =====
        t_track_start = cv2.getTickCount()

        # Health-stat: record a "detection tick" whenever inference produced
        # at least one bounding box this frame. Drives the dashboard's
        # "detect/s" column.
        if self.inference_ran and len(detections) > 0:
            self._health_detect_times.append(now_wall)

        # Update MOT (even if no detections, to run predictions)
        self.mot.update(detections, confidences)
        
        # Check for newly confirmed tracks and trigger alerts
        for track in self.mot.tracks:
            if track.state == track.CONFIRMED and track.track_id not in self.alerted_track_ids:
                self.alerted_track_ids.add(track.track_id)
                if self.on_track_confirmed:
                    self.on_track_confirmed(track.track_id)
                # Always start logging locally (screenshots and CSV work even without server)
                self.track_logger_manager.start_logging(track.track_id)
        
        # AUTO-LOCK: If no target is locked, lock the first confirmed track immediately
        if self.mot.active_track_id is None:
            confirmed_tracks = [t for t in self.mot.tracks if t.state == t.CONFIRMED]
            if confirmed_tracks:
                self.mot.active_track_id = confirmed_tracks[0].track_id
                logger.info(f"[LOCK] Auto-locked first target: #{confirmed_tracks[0].track_id}")
        
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
            
            # CLEANUP: If locked track no longer exists or is LOST, clear lock
            if current_active_track is None or current_active_track.state == current_active_track.LOST:
                logger.info(f"[LOCK] Target #{self.mot.active_track_id} lost, clearing lock")
                self.mot.active_track_id = None
                current_active_track = None

        # === DYNAMIC THREAT ASSESSMENT ===
        # Detect scenario
        self.current_scenario = SwarmAwareThreatAnalyzer.detect_scenario(all_tracks)
        
        # Log scenario changes
        if self.current_scenario != self.previous_scenario:
            confirmed_count = len([t for t in all_tracks if t.state == t.CONFIRMED])
            scenario_names = {0: "SINGLE", 1: "SMALL_GROUP", 2: "MEDIUM_SWARM", 3: "LARGE_SWARM"}
            logger.info(f"[SCENARIO] {scenario_names[self.previous_scenario]} -> {scenario_names[self.current_scenario]} ({confirmed_count} drones)")
            self.previous_scenario = self.current_scenario

        # Dynamic re-assessment
        active_track, switched = SwarmAwareThreatAnalyzer.dynamic_reassessment(
            current_active_track,
            all_tracks,
            self.frame_count,
            self.reassess_timer,
            self.current_scenario,
            self.lock_age_frames
        )

        # Anti-flap: require repeated reassessment agreement before switching.
        if switched and current_active_track and active_track and active_track.track_id != current_active_track.track_id:
            if self.pending_switch_track_id == active_track.track_id:
                self.pending_switch_count += 1
            else:
                self.pending_switch_track_id = active_track.track_id
                self.pending_switch_count = 1

            if self.pending_switch_count < config.SWITCH_CONFIRMATIONS_REQUIRED:
                switched = False
                active_track = current_active_track
        else:
            # Only reset counter when a reassessment actually fired and said MAINTAIN.
            # Without this guard, the counter resets every non-reassessment frame (~119/120 frames),
            # making SWITCH_CONFIRMATIONS_REQUIRED > 1 impossible to satisfy.
            if self.reassess_timer.last_assessment_frame == self.frame_count:
                self.pending_switch_track_id = None
                self.pending_switch_count = 0

        if switched:
            self.pending_switch_track_id = None
            self.pending_switch_count = 0
            self.mot.active_track_id = active_track.track_id if active_track else None
            self.mot.switch_cooldown_counter = 0
            self.lock_start_frame = self.frame_count
            self.lock_age_frames = 0
            
            # CRITICAL: Reset PID and FF on switch
            self.pid_pan.reset()
            self.pid_tilt.reset()
            self._ff_vx = 0.0
            self._ff_vy = 0.0

            # Update last locked pos immediately
            if active_track:
                self.last_locked_pos = (active_track.kf.x[0], active_track.kf.x[1])
            else:
                self.last_locked_pos = None
        else:
            self.mot.switch_cooldown_counter += 1
            self.lock_age_frames += 1
        # Update last locked pos if we have a track
            if active_track:
                self.last_locked_pos = (active_track.kf.x[0], active_track.kf.x[1])
        
        t_track_end = cv2.getTickCount()
        self.last_track_ms = (t_track_end - t_track_start) * 1000.0 / cv2.getTickFrequency()

        # === COMPUTE THREAT SCORES FOR ALL TRACKS (EMA-smoothed for stability) ===
        for t in all_tracks:
            if t.state == t.CONFIRMED:
                is_current = (t == active_track)
                raw_score = SwarmAwareThreatAnalyzer.calculate_advanced_threat_score(
                    t, all_tracks, self.current_scenario,
                    is_current, self.lock_age_frames if is_current else 0
                )
                prev = getattr(t, 'threat_score', raw_score)
                # Asymmetric EMA: quick rise, slow decay (matches reassessment logic)
                _alpha = config.THREAT_SCORE_RISE_ALPHA if raw_score >= prev else config.THREAT_SCORE_DECAY_ALPHA
                t.threat_score = _alpha * raw_score + (1.0 - _alpha) * prev
            else:
                t.threat_score = 0.0

        # Visualize FIRST (so _last_annotated_frame is current for FP screenshots)
        t_draw_start = cv2.getTickCount()
        annotated = frame.copy()

        annotated = self._visualize_tracks(annotated, all_tracks, active_track)

        if active_track:
             # Send telemetry periodically
             self._calculate_and_send_telemetry(active_track)

        # Calculate Threat Zone (for HUD)
        threat_zone = None
        if active_track:
             center_x = config.FRAME_WIDTH / 2
             center_y = config.FRAME_HEIGHT / 2
             try:
                 threat_zone = SwarmAwareThreatAnalyzer.classify_threat_zone(active_track, center_x, center_y)
             except:
                 pass

        draw_hud(annotated, active_track, all_tracks, roi_active=roi_active, inference_ms=self.last_inference_ms, current_sector_name=self.current_sector_name, threat_zone=threat_zone)

        annotated = self._control_servo(annotated, active_track, all_tracks)

        t_draw_end = cv2.getTickCount()
        self.last_draw_ms = (t_draw_end - t_draw_start) * 1000.0 / cv2.getTickFrequency()

        # Store annotated frame for false-positive screenshots (now current, not stale)
        self._last_annotated_frame = annotated

        # --- STATUS REPORTING TO SERVER (LOCKED/PASSIVE/LOST) ---
        # LOCKED = the actively tracked target (red border)
        # PASSIVE = other confirmed tracks (green boxes)
        # LOST = track no longer being tracked
        if self.on_tracking_status:
            current_track_ids = set(t.track_id for t in all_tracks)
            active_track_id = self.mot.active_track_id  # The locked target

            # 2 Hz emit cadence for non-locked confirmed tracks (vs 6 Hz for the
            # locked one in _calculate_and_send_telemetry). Bounds bandwidth on
            # multi-target scenes.
            self._passive_telemetry_frame_counter += 1
            passive_emit_tick = (self._passive_telemetry_frame_counter % 15 == 0)

            # Report current track statuses
            for track in all_tracks:
                if track.state == track.CONFIRMED:
                    # Distinguish between locked and passive
                    is_locked = (track.track_id == active_track_id)
                    new_status = 'LOCKED' if is_locked else 'PASSIVE'

                    # Estimate 3D state for locked track always, passive tracks periodically.
                    should_update_3d = is_locked or (
                        config.PASSIVE_3D_UPDATE_INTERVAL <= 1 or
                        (self.frame_count % config.PASSIVE_3D_UPDATE_INTERVAL == 0)
                    )
                    if should_update_3d and track.box_width > 0:
                        REAL_DRONE_WIDTH = config.KNOWN_WIDTH_CM / 100.0  # Convert cm to meters
                        FOCAL_LENGTH = config.FOCAL_LENGTH
                        CENTER_X = config.FRAME_WIDTH / 2
                        CENTER_Y = config.FRAME_HEIGHT / 2
                        dt = 1.0 / 30.0
                        track.estimate_3d_state(FOCAL_LENGTH, REAL_DRONE_WIDTH, CENTER_X, CENTER_Y, dt=dt)

                    # Ship world-frame geolocation for PASSIVE tracks too, so the
                    # central minimap can plot every detected drone, not just the
                    # engaged primary. LOCKED emits at 6 Hz from the active path.
                    if (not is_locked) and passive_emit_tick and self.on_active_telemetry:
                        distance = track.real_pos[2] if hasattr(track, 'real_pos') else 0.0
                        if distance > 0.0:
                            try:
                                self.on_active_telemetry(
                                    track.track_id,
                                    float(self.pan_angle),
                                    float(self.tilt_angle),
                                    float(track.get_azimuth()),
                                    float(track.get_elevation()),
                                    float(distance),
                                    "PASSIVE",
                                )
                            except Exception as e:
                                logger.warning(f"Passive telemetry callback failed: {e}")

                    # Update track logger with telemetry (for ALL confirmed tracks)
                    self.track_logger_manager.update(track, self.frame_count, is_locked=is_locked)

                elif track.state == track.LOST:
                    new_status = 'LOST'
                else:
                    continue  # TENTATIVE, DELETED - skip

                # Send if status changed
                old_status = self.track_status_cache.get(track.track_id)
                if old_status != new_status:
                    logger.info(f"[STATUS] Track #{track.track_id}: {old_status} -> {new_status}")
                    self.on_tracking_status(track.track_id, new_status)
                    self.track_status_cache[track.track_id] = new_status

                    # Finalize log when track goes LOST
                    if new_status == 'LOST':
                        # Determine if it was lost as LOCKED or PASSIVE
                        was_locked = (old_status == 'LOCKED')
                        reason = 'LOST_LOCKED' if was_locked else 'LOST_PASSIVE'

                        # Finalize track log
                        summary = self.track_logger_manager.finalize(track.track_id, reason)
                        if summary:
                            # Screenshot for potential false positives
                            if summary.get('possible_false') and self._last_annotated_frame is not None:
                                import base64, os
                                from pathlib import Path
                                fp_dir = Path("./vlogs")
                                fp_dir.mkdir(parents=True, exist_ok=True)
                                screenshot_path = str(fp_dir / f"{config.DEVICE_NAME}_fp_{track.track_id}_{int(time.time())}.jpg")
                                cv2.imwrite(screenshot_path, self._last_annotated_frame)
                                summary['screenshot_path'] = screenshot_path
                                summary['screenshot_filename'] = os.path.basename(screenshot_path)
                                with open(screenshot_path, 'rb') as _f:
                                    summary['screenshot_data'] = base64.b64encode(_f.read()).decode('utf-8')
                                logger.info(f"[FP] Saved false-positive screenshot: {screenshot_path}")
                            if self.on_track_log_complete:
                                self.on_track_log_complete(summary)

            # Handle tracks that disappeared (deleted/merged)
            for track_id in list(self.track_status_cache.keys()):
                if track_id not in current_track_ids:
                    old_status = self.track_status_cache.get(track_id)
                    logger.info(f"[STATUS] Track #{track_id}: {old_status} -> LOST (deleted)")
                    self.on_tracking_status(track_id, 'LOST')
                    del self.track_status_cache[track_id]
                    if self.on_track_deleted:
                        try:
                            self.on_track_deleted(track_id)
                        except Exception as e:
                            logger.warning(f"on_track_deleted callback failed: {e}")

                    # Finalize track log
                    was_locked = (old_status == 'LOCKED')
                    reason = 'DELETED_LOCKED' if was_locked else 'DELETED_PASSIVE'
                    summary = self.track_logger_manager.finalize(track_id, reason)
                    if summary:
                        if summary.get('possible_false') and self._last_annotated_frame is not None:
                            import base64, os
                            from pathlib import Path
                            fp_dir = Path("./vlogs")
                            fp_dir.mkdir(parents=True, exist_ok=True)
                            screenshot_path = str(fp_dir / f"{config.DEVICE_NAME}_fp_{track_id}_{int(time.time())}.jpg")
                            cv2.imwrite(screenshot_path, self._last_annotated_frame)
                            summary['screenshot_path'] = screenshot_path
                            summary['screenshot_filename'] = os.path.basename(screenshot_path)
                            with open(screenshot_path, 'rb') as _f:
                                summary['screenshot_data'] = base64.b64encode(_f.read()).decode('utf-8')
                            logger.info(f"[FP] Saved false-positive screenshot: {screenshot_path}")
                        if self.on_track_log_complete:
                            self.on_track_log_complete(summary)

        return annotated, self.pan_angle, self.tilt_angle