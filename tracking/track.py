import math
import time
import numpy as np
from filterpy.kalman import KalmanFilter
from collections import deque

import config


class Track:
    # Individual tracked object with Kalman filter.
    # Optimized for stability.
    
    # States
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    DELETED = 3
    
    # Speed thresholds
    # Velocities are in px/s (measured dt); thresholds were tuned in px/frame at 30 fps.
    FAST_MOTION_THRESHOLD = 12.0 * config.TARGET_FPS
    AGGRESSIVE_MODE_THRESHOLD = 18.0 * config.TARGET_FPS
    
    # SCREEN BOUNDARIES (with margin)
    SCREEN_MARGIN = 100  # pixels outside screen before considering "escaped"
    
    # DELETION PARAMETERS
    # LOST tracks survive on Kalman prediction!
    TENTATIVE_MAX_AGE = 3        # Unconfirmed: 0.1 sec
    LOST_MAX_AGE_SLOW = 30       # Slow tracks: 1 sec
    LOST_MAX_AGE_FAST = 30       # Fast tracks: 1 sec
    LOST_THRESHOLD_SLOW = 20     # Mark lost after 20 frames (was 10)
    LOST_THRESHOLD_FAST = 30     # Fast objects: 30 frames (was 20)
    
    # CONFIRMATION PARAMETERS
    MIN_HITS_TO_CONFIRM = 4      # Need 4 hits (Fast but stable)
    MIN_HITS_FAST = 3            # Fast objects confirm with 3
    
    def __init__(self, track_id, initial_detection, initial_conf):
        self.track_id = track_id
        self.state = self.TENTATIVE
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.consecutive_misses = 0
        self.last_update_frame = 0
        
        # Kalman Filter
        # burası önemli
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        x1, y1, x2, y2 = initial_detection
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        self.kf.x = np.array([cx, cy, 0.0, 0.0])
        self.kf.F = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], dtype=float)
        self.kf.H = np.array([[1,0,0,0], [0,1,0,0]], dtype=float)
        self.kf.P *= 500.0 
        
        # TUNED for stability
        # TUNED for RESPONSIVENESS (User request: fix lag/memory issue)
        # Trust detection MORE (Lower R) - was 8
        self.kf.R = np.array([[2,0], [0,2]], dtype=float) 
        
        # Process noise: position=10, velocity=10 (was 50 — caused jittery velocity estimates)
        self.kf.Q = np.array([
            [10, 0, 0, 0],
            [0, 10, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 10]
        ], dtype=float)
        
        self.base_Q = self.kf.Q.copy()

        # Timing for dt-aware predict
        self.last_predict_time = None

        # Metadata
        self.last_detection = initial_detection
        self.last_confidence = initial_conf
        self.trajectory = deque(maxlen=30)
        self.trajectory.append((cx, cy))
        
        # Speed tracking
        self.speed_history = deque(maxlen=15)
        self.speed_history.append(0.0)
        
        # Box size tracking (for swarm detection stability)
        self.box_width = x2 - x1
        self.box_height = y2 - y1
        # Width history for depth-estimate median filter. Monocular depth is
        # depth ∝ 1/box_width, so a 1-pixel detector spike on a 30-px-wide
        # drone bbox is ~3% depth error → ~5 m position shift at 150 m range.
        # A short rolling median absorbs detector-level shot noise without
        # adding visible lag on a moving target (window = 5 frames @ 30 fps
        # ≈ 167 ms). See estimate_3d_state() for the usage.
        self._width_history = deque(maxlen=5)
        self._width_history.append(float(x2 - x1))
        
        # Threat scoring
        self.threat_score = 0.0
        self.nearby_drone_count = 0
        
        # Spam prevention
        self.last_motion_log_frame = 0
        
        # --- 3D KINEMATIC FILTER (Secondary KF) ---
        # State: [X, Y, Z, Vx, Vy, Vz]
        self.kf3d = KalmanFilter(dim_x=6, dim_z=3)
        
        # State Transition Function (Constant Velocity Model)
        # x = x + vx*dt, y = y + vy*dt, z = z + vz*dt
        self.kf3d.F = np.eye(6)
        self.kf3d.F[0, 3] = 1.0  # x <- vx
        self.kf3d.F[1, 4] = 1.0  # y <- vy
        self.kf3d.F[2, 5] = 1.0  # z <- vz
        
        # Measurement Function (We measure X, Y, Z directly from pinhole)
        self.kf3d.H = np.zeros((3, 6))
        self.kf3d.H[0, 0] = 1.0  # Measure X
        self.kf3d.H[1, 1] = 1.0  # Measure Y
        self.kf3d.H[2, 2] = 1.0  # Measure Z
        
        # Initial Covariance (Uncertainty)
        self.kf3d.P *= 1000.0  # High initial uncertainty
        
        # Process Noise (Q) - How much we trust physics vs "jerks"
        # Drones can change speed, but not instantly.
        self.kf3d.Q = np.eye(6) * 0.1
        self.kf3d.Q[3:, 3:] *= 2.0 # Higher noise in velocity
        
        # Measurement Noise (R) - Starts high, adaptive later
        self.kf3d.R = np.eye(3) * 10.0

        self.base_Q_3d = self.kf3d.Q.copy()
    
    def get_current_speed(self):
        vx, vy = self.kf.x[2], self.kf.x[3]
        return math.sqrt(vx**2 + vy**2)
    
    def get_average_speed(self):
        return np.mean(self.speed_history) if self.speed_history else 0.0
    
    def is_fast_moving(self):
        return self.get_average_speed() > self.FAST_MOTION_THRESHOLD
    
    def is_aggressive_motion(self):
        return self.get_average_speed() > self.AGGRESSIVE_MODE_THRESHOLD
    
    # --- MONOCULAR 3D ESTIMATION (Scientific Implementation) ---
    def estimate_3d_state(self, focal_length, real_width, center_x, center_y, dt=1.0/30.0):
        """
        Estimates 3D kinematic state using a secondary Extended Kalman Filter approach.
        Uses Distance-Dependent Measurement Noise to handle monocular depth uncertainty.
        
        State: [x, y, z, vx, vy, vz]
        """
        # 1. Raw Pinhole Measurement
        if self.box_width < 1: return

        # Depth uses MEDIAN-filtered width, not the EMA copy. Median is robust
        # to single-frame detector spikes — a 1-px spike on a 30-px box flips
        # ~3 % depth = ~5 m at 150 m range, visible as map jitter. Median over
        # the last 5 frames absorbs that without lagging real motion.
        width_for_depth = float(np.median(self._width_history)) if len(self._width_history) > 0 else self.box_width
        if width_for_depth < 1: width_for_depth = self.box_width
        z_meas = (focal_length * real_width) / width_for_depth
        
        # Project 2D center to 3D ray
        current_cx, current_cy = self.kf.x[0], self.kf.x[1]
        x_meas = ((current_cx - center_x) * z_meas) / focal_length
        y_meas = ((current_cy - center_y) * z_meas) / focal_length
        
        # 2. Initialize if first run
        if not hasattr(self, 'real_pos'):
            self.kf3d.x = np.array([x_meas, y_meas, z_meas, 0.0, 0.0, 0.0])
            self.real_pos = np.array([x_meas, y_meas, z_meas])
            self.real_vel = np.array([0.0, 0.0, 0.0])
            return

        # 3. FIXED Measurement Noise (Trust the inputs!)
        # Removed adaptive scaling that was killing speed at range.
        
        var_xy = 0.05 ** 2
        var_z = 0.2 ** 2  # Trust Z-measurements significantly more
        
        self.kf3d.R = np.diag([var_xy, var_xy, var_z])
        
        # 4. Process Noise (Responsive Physics)
        # Removed ZUPT "freezing" - always allow motion updates

        # Scale Q by dt relative to nominal 1/30s step
        dt_scale_3d = dt / (1.0 / 30.0)
        self.kf3d.Q = self.base_Q_3d * dt_scale_3d

        # 5. Update F with actual dt, then Predict & Update
        self.kf3d.F[0, 3] = dt
        self.kf3d.F[1, 4] = dt
        self.kf3d.F[2, 5] = dt

        self.kf3d.predict()
        
        # Store previous position for Direct Speed Calc
        prev_pos = self.kf3d.x[:3].copy()
        
        self.kf3d.update(np.array([x_meas, y_meas, z_meas]))
        
        # 6. Extract State & Safety Clamping
        # Prevent Z from going negative (behind camera), which flips azimuth to +/- 180
        if self.kf3d.x[2] < 0.5:
            self.kf3d.x[2] = 0.5
            self.kf3d.x[5] = 0.0 # Kill Z velocity if we hit the "screen"
            
        self.real_pos = self.kf3d.x[:3]
        self.real_vel = self.kf3d.x[3:]
        
        # 7. Sliding Window Speed Calculation (Stable)
        # ------------------------------------------------------------------
        # Problem: Frame-to-frame depth noise gets amplified by FPS (30x or 60x).
        # Solution: Calculate distance traveled over ~0.25 seconds (8 frames).
        
        if not hasattr(self, 'pos_history'):
            self.pos_history = [] # Simple list checks are fast enough for small N
            self.avg_speed = 0.0

        # Add current position to history
        self.pos_history.append(self.real_pos.copy())
        
        # Keep window size (e.g., 8 frames @ 30fps ~= 0.26s)
        WINDOW_SIZE = 8
        if len(self.pos_history) > WINDOW_SIZE:
            self.pos_history.pop(0)

        # Calculate speed over the window
        if len(self.pos_history) >= 2:
            # Distance from Oldest point to Current point
            oldest_pos = self.pos_history[0]
            current_pos = self.pos_history[-1]
            
            dist_moved = np.linalg.norm(current_pos - oldest_pos)
            
            # Time elapsed = (NumSteps) * dt
            time_window = (len(self.pos_history) - 1) * dt
            
            if time_window > 0:
                window_speed = dist_moved / time_window
                
                # Debug Speed Calculation
                # print(f"[SPEED DEBUG] Dist: {dist_moved:.2f}m | Time: {time_window:.3f}s | Raw: {window_speed:.2f}m/s")
                
                # Apply EMA Smoothing (Alpha 0.1 for stability, 0.3 for responsiveness)
                # Lower alpha because the window itself provides responsiveness
                alpha = 0.15 
                self.avg_speed = (alpha * window_speed) + ((1.0 - alpha) * self.avg_speed)
            else:
                self.avg_speed = 0.0
        else:
            self.avg_speed = 0.0
        
    def get_real_speed(self):
        """Returns direct averaged speed in m/s"""
        if hasattr(self, 'avg_speed'):
            # User request: "I want something real something not fake"
            # Removed the 40.0 cap.
            return self.avg_speed
        return 0.0
        
    def get_azimuth(self):
        """Returns azimuth angle in degrees relative to camera center (Smoothed)"""
        if hasattr(self, 'real_pos'):
            # atan2(x, z) gives angle from forward vector
            raw_azimuth = math.degrees(math.atan2(self.real_pos[0], self.real_pos[2]))

            # Initialize smoother if needed
            if not hasattr(self, 'smoothed_azimuth'):
                self.smoothed_azimuth = raw_azimuth

            # EMA Smoothing (Alpha 0.2 = heavy smoothing)
            self.smoothed_azimuth = (0.2 * raw_azimuth) + (0.8 * self.smoothed_azimuth)

            # Clamp to FOV (+/- 60 degrees) to prevent "100 degree" glitches
            return max(-60.0, min(60.0, self.smoothed_azimuth))

        return 0.0

    def get_elevation(self):
        """Returns optical elevation in degrees relative to camera horizon (Smoothed).
        Positive = target above camera center; negative = below."""
        if hasattr(self, 'real_pos'):
            x, y, z = self.real_pos[0], self.real_pos[1], self.real_pos[2]
            # real_pos[1] is image-Y (positive DOWNWARD). Flip for elevation.
            horiz = math.sqrt(x * x + z * z)
            raw_elev = math.degrees(math.atan2(-y, horiz)) if horiz > 1e-6 else 0.0

            if not hasattr(self, 'smoothed_elevation'):
                self.smoothed_elevation = raw_elev

            self.smoothed_elevation = (0.2 * raw_elev) + (0.8 * self.smoothed_elevation)
            return max(-45.0, min(45.0, self.smoothed_elevation))

        return 0.0
    
    def is_outside_screen(self):
        sx, sy = self.kf.x[0], self.kf.x[1]
        
        min_x = -self.SCREEN_MARGIN
        max_x = config.FRAME_WIDTH + self.SCREEN_MARGIN
        min_y = -self.SCREEN_MARGIN
        max_y = config.FRAME_HEIGHT + self.SCREEN_MARGIN
        
        return sx < min_x or sx > max_x or sy < min_y or sy > max_y
    
    def predict(self):
        current_speed = self.get_current_speed()
        self.speed_history.append(current_speed)

        # --- Measure actual dt ---
        now = time.monotonic()
        if self.last_predict_time is None:
            dt = 1.0 / 30.0  # First call: assume 30fps
        else:
            dt = now - self.last_predict_time
            dt = max(0.001, min(dt, 0.2))  # Clamp to [1ms, 200ms]
        self.last_predict_time = now

        # --- Update F with real dt ---
        self.kf.F[0, 2] = dt  # x += vx * dt
        self.kf.F[1, 3] = dt  # y += vy * dt

        # VELOCITY DAMPING - GRADUAL for moving, AGGRESSIVE for stationary
        if self.time_since_update > 3:
            avg_speed = self.get_average_speed()

            if avg_speed < 3.0 * config.TARGET_FPS:
                # STATIONARY: Very aggressive damping - stay in place!
                damping = 0.5 ** self.time_since_update
                damping = max(damping, 0.01)  # Almost zero
            else:
                # MOVING: Gradual damping
                damping = 0.95 ** (self.time_since_update - 3)
                damping = max(damping, 0.3)

            self.kf.x[2] *= damping
            self.kf.x[3] *= damping

        if self.hits < 10 and self.time_since_update > 0:
            self.kf.x[2] *= 0.1
            self.kf.x[3] *= 0.1

        # --- Adaptive Q based on motion, scaled by dt ---
        # Scale factor: dt normalised against the nominal 1/30s step
        dt_scale = dt / (1.0 / 30.0)

        if self.is_aggressive_motion():
            self.kf.Q = self.base_Q * 3.0 * dt_scale
            if self.age - self.last_motion_log_frame > 30:
                print(f"[MOTION] ID {self.track_id} AGGRESSIVE (speed: {current_speed:.1f})")
                self.last_motion_log_frame = self.age
        elif self.is_fast_moving():
            self.kf.Q = self.base_Q * 2.0 * dt_scale
        else:
            # STATIONARY: Very low Q - trust position, predict no movement
            if self.time_since_update == 0:
                self.kf.Q = self.base_Q * 0.3 * dt_scale
            else:
                self.kf.Q = self.base_Q * 0.1 * dt_scale

        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection, confidence, frame_count=0):
        x1, y1, x2, y2 = detection
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Update box size (smoothed)
        new_w = x2 - x1
        new_h = y2 - y1
        self.box_width = 0.7 * self.box_width + 0.3 * new_w
        self.box_height = 0.7 * self.box_height + 0.3 * new_h
        # Raw width into the depth-median filter (not the EMA copy — median
        # is robust to single-frame spikes that would still bleed through
        # the 0.3-alpha EMA above).
        self._width_history.append(float(new_w))
        
        # Adaptive R based on confidence
        adaptive_R = self.kf.R.copy()
        
        # Lower confidence = higher R (trust detection less)
        confidence_factor = max(0.5, 2.0 - confidence * 1.5)
        adaptive_R *= confidence_factor
        
        # Fast moving = trust detection more
        if self.is_fast_moving():
            adaptive_R *= 0.8
        
        self.kf.update(z=np.array([cx, cy]), R=adaptive_R)
        
        self.last_detection = detection
        self.last_confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.consecutive_misses = 0
        self.last_update_frame = frame_count
        
        # Update trajectory
        sx, sy = self.kf.x[0], self.kf.x[1]
        self.trajectory.append((sx, sy))
        
        # State transitions
        if self.state == self.TENTATIVE:
            required_hits = self.MIN_HITS_FAST if self.is_fast_moving() else self.MIN_HITS_TO_CONFIRM
            if self.hits >= required_hits:
                self.state = self.CONFIRMED
                # Fix: Clear warmup trajectory to prevent "jumping" lines
                self.trajectory.clear()
                self.trajectory.append((cx, cy))
                print(f"[TRACK] ID {self.track_id} CONFIRMED (hits:{self.hits})")
        elif self.state == self.LOST:
            # Only log if was lost for significant time
            if self.time_since_update > 10:
                print(f"[TRACK] ID {self.track_id} RE-ACQUIRED (was lost {self.time_since_update}f)")
                
            # Fix jump artifact: Clear history if lost for too long
            if self.time_since_update > 5:
                self.trajectory.clear() # Don't draw line from old pos
                
                # Reset velocity if gap is huge (prevent shooting off)
                if self.time_since_update > 15:
                    self.kf.x[2] *= 0.1
                    self.kf.x[3] *= 0.1

            self.state = self.CONFIRMED
    
    def mark_missed(self):
        self.consecutive_misses += 1
        
        if self.state == self.CONFIRMED:
            lost_threshold = self.LOST_THRESHOLD_FAST if self.is_fast_moving() else self.LOST_THRESHOLD_SLOW
            
            if self.time_since_update > lost_threshold:
                self.state = self.LOST
                # Only log first time (not spam on re-acquire cycles)
                if self.consecutive_misses == self.time_since_update:
                    print(f"[TRACK] ID {self.track_id} LOST (missed:{self.time_since_update}f)")
    
    def is_deleted(self):
        # Check if track should be deleted.
        sx, sy = self.kf.x[0], self.kf.x[1]
        if sx < -200 or sx > config.FRAME_WIDTH + 200:
            return True
        if sy < -200 or sy > config.FRAME_HEIGHT + 200:
            return True
        
        # IMMEDIATE DELETE: Kalman diverged (unrealistic state)
        if self.get_current_speed() > 150 * config.TARGET_FPS:  # Way too fast
            return True
        
        # Delete tentative tracks that never confirmed
        if self.state == self.TENTATIVE and self.time_since_update > self.TENTATIVE_MAX_AGE:
            return True
        
        # LOST tracks get extended lifetime based on motion history
        if self.state == self.LOST:
            # Fast-moving tracks get MORE time (they tend to return)
            if self.is_fast_moving():
                return self.time_since_update > self.LOST_MAX_AGE_FAST
            else:
                return self.time_since_update > self.LOST_MAX_AGE_SLOW
        
        # CONFIRMED tracks that have been lost too long
        # But give them more time than before!
        if self.state == self.CONFIRMED and self.time_since_update > self.LOST_MAX_AGE_SLOW:
            return True
        
        return False
    
    def get_predicted_box(self):
        sx, sy = self.kf.x[0], self.kf.x[1]
        
        # Use smoothed box size
        w = self.box_width
        h = self.box_height
        
        return (int(sx - w/2), int(sy - h/2), int(sx + w/2), int(sy + h/2))
    
    def distance_to(self, other_track):
        sx, sy = self.kf.x[0], self.kf.x[1]
        ox, oy = other_track.kf.x[0], other_track.kf.x[1]
        return math.sqrt((sx - ox)**2 + (sy - oy)**2)