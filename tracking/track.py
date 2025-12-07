# tracking/track.py
"""
Track class v2: Individual tracked object state.
Key changes:
1. FASTER deletion of ghost tracks
2. Screen boundary checking
3. Velocity damping for stationary objects
4. Better state transitions for swarm scenarios
"""
import math
import numpy as np
from filterpy.kalman import KalmanFilter
from collections import deque

import config


class Track:
    """
    Individual tracked object with Kalman filter.
    Optimized for swarm tracking stability.
    """
    
    # States
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    DELETED = 3
    
    # Speed thresholds
    FAST_MOTION_THRESHOLD = 12.0
    AGGRESSIVE_MODE_THRESHOLD = 18.0
    
    # SCREEN BOUNDARIES (with margin)
    SCREEN_MARGIN = 100  # pixels outside screen before considering "escaped"
    
    # DELETION PARAMETERS (FROM WORKING 16-17/11 SYSTEM)
    # Key insight: LOST tracks should survive on Kalman prediction!
    TENTATIVE_MAX_AGE = 3        # Unconfirmed: 0.1 sec (Strict noise cleanup)
    LOST_MAX_AGE_SLOW = 60       # Slow tracks: 2 sec before delete
    LOST_MAX_AGE_FAST = 90       # Fast tracks: 3 sec (they WILL return!)
    LOST_THRESHOLD_SLOW = 10     # Mark as lost after 10 frames
    LOST_THRESHOLD_FAST = 20     # Fast objects: 20 frames (more tolerance)
    
    # CONFIRMATION PARAMETERS
    MIN_HITS_TO_CONFIRM = 4      # Need 4 hits (Fast but stable)
    MIN_HITS_FAST = 3            # Fast objects confirm with 3
    
    def __init__(self, track_id, initial_detection, initial_conf):
        self.track_id = track_id
        self.state = self.TENTATIVE
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.consecutive_misses = 0  # NEW: track consecutive misses
        self.last_update_frame = 0   # NEW: for adaptive inference frame skipping
        
        # Kalman Filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        x1, y1, x2, y2 = initial_detection
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        self.kf.x = np.array([cx, cy, 0.0, 0.0])
        self.kf.F = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], dtype=float)
        self.kf.H = np.array([[1,0,0,0], [0,1,0,0]], dtype=float)
        self.kf.P *= 500.0  # Reduced from 1000 - less uncertainty
        
        # TUNED for stability
        self.kf.R = np.array([[8,0], [0,8]], dtype=float)  # Increased - trust detection less
        self.kf.Q = np.array([
            [5, 0, 0, 0],    # Reduced position noise
            [0, 5, 0, 0],
            [0, 0, 20, 0],   # Reduced velocity noise
            [0, 0, 0, 20]
        ], dtype=float)
        
        self.base_Q = self.kf.Q.copy()
        
        # Metadata
        self.last_detection = initial_detection
        self.last_confidence = initial_conf
        self.trajectory = deque(maxlen=30)
        self.trajectory.append((cx, cy))
        
        # Speed tracking
        self.speed_history = deque(maxlen=5)
        self.speed_history.append(0.0)
        
        # Box size tracking (for swarm detection stability)
        self.box_width = x2 - x1
        self.box_height = y2 - y1
        
        # Threat scoring
        self.threat_score = 0.0
        self.nearby_drone_count = 0
        
        # Spam prevention
        self.last_motion_log_frame = 0
    
    def get_current_speed(self):
        vx, vy = self.kf.x[2], self.kf.x[3]
        return math.sqrt(vx**2 + vy**2)
    
    def get_average_speed(self):
        return np.mean(self.speed_history) if self.speed_history else 0.0
    
    def is_fast_moving(self):
        return self.get_average_speed() > self.FAST_MOTION_THRESHOLD
    
    def is_aggressive_motion(self):
        return self.get_average_speed() > self.AGGRESSIVE_MODE_THRESHOLD
    
    def is_outside_screen(self):
        """Check if track has drifted outside screen boundaries"""
        sx, sy = self.kf.x[0], self.kf.x[1]
        
        min_x = -self.SCREEN_MARGIN
        max_x = config.FRAME_WIDTH + self.SCREEN_MARGIN
        min_y = -self.SCREEN_MARGIN
        max_y = config.FRAME_HEIGHT + self.SCREEN_MARGIN
        
        return sx < min_x or sx > max_x or sy < min_y or sy > max_y
    
    def predict(self):
        """Kalman prediction with adaptive Q and GRADUAL velocity damping"""
        current_speed = self.get_current_speed()
        self.speed_history.append(current_speed)
        
        # VELOCITY DAMPING - GRADUAL for moving, AGGRESSIVE for stationary
        if self.time_since_update > 3:
            avg_speed = self.get_average_speed()
            
            if avg_speed < 3.0:
                # STATIONARY: Very aggressive damping - stay in place!
                damping = 0.5 ** self.time_since_update
                damping = max(damping, 0.01)  # Almost zero
            else:
                # MOVING: Gradual damping
                damping = 0.95 ** (self.time_since_update - 3)
                damping = max(damping, 0.3)
            
            self.kf.x[2] *= damping
            self.kf.x[3] *= damping
        
        # SAFETY: If track is young (<10 hits) and we are predicting (missing measurements),
        # kill velocity to prevent "shooting off" into space based on noise.
        if self.hits < 10 and self.time_since_update > 0:
            self.kf.x[2] *= 0.1
            self.kf.x[3] *= 0.1
        
        # Adaptive Q based on motion
        if self.is_aggressive_motion():
            self.kf.Q = self.base_Q * 3.0
            if self.age - self.last_motion_log_frame > 30:
                print(f"[MOTION] ID {self.track_id} AGGRESSIVE (speed: {current_speed:.1f})")
                self.last_motion_log_frame = self.age
        elif self.is_fast_moving():
            self.kf.Q = self.base_Q * 2.0
        else:
            # STATIONARY: Very low Q - trust position, predict no movement
            if self.time_since_update == 0:
                self.kf.Q = self.base_Q * 0.3  # Smooth when detected
            else:
                self.kf.Q = self.base_Q * 0.1  # Very smooth when predicting stationary
        
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection, confidence, frame_count=0):
        """Update with new detection"""
        x1, y1, x2, y2 = detection
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Update box size (smoothed)
        new_w = x2 - x1
        new_h = y2 - y1
        self.box_width = 0.7 * self.box_width + 0.3 * new_w
        self.box_height = 0.7 * self.box_height + 0.3 * new_h
        
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
        self.consecutive_misses = 0  # Reset miss counter
        self.last_update_frame = frame_count # Update frame count
        
        # Update trajectory
        sx, sy = self.kf.x[0], self.kf.x[1]
        self.trajectory.append((sx, sy))
        
        # State transitions
        if self.state == self.TENTATIVE:
            required_hits = self.MIN_HITS_FAST if self.is_fast_moving() else self.MIN_HITS_TO_CONFIRM
            if self.hits >= required_hits:
                self.state = self.CONFIRMED
                print(f"[TRACK] ID {self.track_id} CONFIRMED (hits:{self.hits})")
        elif self.state == self.LOST:
            # Only log if was lost for significant time
            if self.time_since_update > 10:
                print(f"[TRACK] ID {self.track_id} RE-ACQUIRED (was lost {self.time_since_update}f)")
            self.state = self.CONFIRMED
    
    def mark_missed(self):
        """Called when no detection matched this track"""
        self.consecutive_misses += 1
        
        if self.state == self.CONFIRMED:
            lost_threshold = self.LOST_THRESHOLD_FAST if self.is_fast_moving() else self.LOST_THRESHOLD_SLOW
            
            if self.time_since_update > lost_threshold:
                self.state = self.LOST
                # Only log first time (not spam on re-acquire cycles)
                if self.consecutive_misses == self.time_since_update:
                    print(f"[TRACK] ID {self.track_id} LOST (missed:{self.time_since_update}f)")
    
    def is_deleted(self):
        """
        Check if track should be deleted.
        Key principle: LOST tracks survive on Kalman prediction!
        Only delete when truly hopeless.
        """
        # IMMEDIATE DELETE: Way outside screen (no hope of return)
        sx, sy = self.kf.x[0], self.kf.x[1]
        if sx < -200 or sx > config.FRAME_WIDTH + 200:
            return True
        if sy < -200 or sy > config.FRAME_HEIGHT + 200:
            return True
        
        # IMMEDIATE DELETE: Kalman diverged (unrealistic state)
        if self.get_current_speed() > 150:  # Way too fast
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
        """Get predicted bounding box"""
        sx, sy = self.kf.x[0], self.kf.x[1]
        
        # Use smoothed box size
        w = self.box_width
        h = self.box_height
        
        return (int(sx - w/2), int(sy - h/2), int(sx + w/2), int(sy + h/2))
    
    def distance_to(self, other_track):
        """Calculate distance to another track (for merging)"""
        sx, sy = self.kf.x[0], self.kf.x[1]
        ox, oy = other_track.kf.x[0], other_track.kf.x[1]
        return math.sqrt((sx - ox)**2 + (sy - oy)**2)