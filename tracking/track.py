"""
Track class: Manages individual tracked object state, Kalman filtering, and lifecycle.
"""
import math
import numpy as np
from filterpy.kalman import KalmanFilter
from collections import deque

import config


class Track:
    """
    Represents a single tracked object with Kalman filter and state management.
    
    States:
        TENTATIVE: New track, needs confirmation
        CONFIRMED: Reliable track
        LOST: No detection for a while
        DELETED: Removed from system
    """
    
    # Track states
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    DELETED = 3
    
    # Speed thresholds (pixels/frame)
    FAST_MOTION_THRESHOLD = 12.0
    AGGRESSIVE_MODE_THRESHOLD = 18.0
    
    def __init__(self, track_id, initial_detection, initial_conf):
        self.track_id = track_id
        self.state = self.TENTATIVE
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        
        # Initialize Kalman Filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        x1, y1, x2, y2 = initial_detection
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        self.kf.x = np.array([cx, cy, 0.0, 0.0])
        self.kf.F = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], dtype=float)
        self.kf.H = np.array([[1,0,0,0], [0,1,0,0]], dtype=float)
        self.kf.P *= 1000.0
        self.kf.R = np.array([[5,0], [0,5]], dtype=float)
        self.kf.Q = np.array([[10,0,0,0], [0,10,0,0], [0,0,50,0], [0,0,0,50]], dtype=float)
        
        self.base_Q = self.kf.Q.copy()
        
        # Track metadata
        self.last_detection = initial_detection
        self.last_confidence = initial_conf
        self.trajectory = deque(maxlen=30)
        self.trajectory.append((cx, cy))
        
        # Speed tracking
        self.speed_history = deque(maxlen=5)
        self.speed_history.append(0.0)
        
        # Threat scoring
        self.threat_score = 0.0
        
    def get_current_speed(self):
        """Calculate current speed magnitude"""
        vx, vy = self.kf.x[2], self.kf.x[3]
        return math.sqrt(vx**2 + vy**2)
    
    def get_average_speed(self):
        """Get smoothed speed over last 5 frames"""
        return np.mean(self.speed_history) if self.speed_history else 0.0
    
    def is_fast_moving(self):
        """Check if track is in fast motion"""
        return self.get_average_speed() > self.FAST_MOTION_THRESHOLD
    
    def is_aggressive_motion(self):
        """Check if track is in very aggressive motion"""
        return self.get_average_speed() > self.AGGRESSIVE_MODE_THRESHOLD
    
    def predict(self):
        """Kalman prediction with motion-adaptive Q matrix"""
        current_speed = self.get_current_speed()
        self.speed_history.append(current_speed)
        
        # Adaptive Q based on motion
        if self.is_aggressive_motion():
            self.kf.Q = self.base_Q * 3.0
            print(f"[MOTION] ID {self.track_id} AGGRESSIVE (speed: {current_speed:.1f})")
        elif self.is_fast_moving():
            self.kf.Q = self.base_Q * 2.0
        else:
            if self.age > 10:
                self.kf.Q = self.base_Q * 0.5
            else:
                self.kf.Q = self.base_Q.copy()
        
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
    def update(self, detection, confidence):
        """Update track with new detection"""
        x1, y1, x2, y2 = detection
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Adaptive R based on confidence and motion
        adaptive_R = self.kf.R.copy()
        adaptive_R *= (1.0 - confidence)
        
        if self.is_fast_moving():
            adaptive_R *= 0.7
        
        self.kf.update(z=np.array([cx, cy]), R=adaptive_R)
        
        self.last_detection = detection
        self.last_confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        
        # Update trajectory
        sx, sy = self.kf.x[0], self.kf.x[1]
        self.trajectory.append((sx, sy))
        
        # State transitions
        if self.state == self.TENTATIVE:
            required_hits = 2 if self.is_fast_moving() else 3
            if self.hits >= required_hits:
                self.state = self.CONFIRMED
                print(f"[TRACK] ID {self.track_id} CONFIRMED")
        elif self.state == self.LOST:
            self.state = self.CONFIRMED
            print(f"[TRACK] ID {self.track_id} RE-ACQUIRED (lost: {self.time_since_update}f)")
    
    def mark_missed(self):
        """Called when no detection matched"""
        if self.state == self.CONFIRMED:
            lost_threshold = 20 if self.is_fast_moving() else 10
            
            if self.time_since_update > lost_threshold:
                self.state = self.LOST
                print(f"[TRACK] ID {self.track_id} LOST (speed: {self.get_average_speed():.1f})")
    
    def is_deleted(self):
        """Check if track should be deleted"""
        if self.state == self.TENTATIVE and self.time_since_update > 10:
            return True
        
        if self.state == self.LOST:
            if self.is_fast_moving():
                return self.time_since_update > 90
            else:
                return self.time_since_update > 60
        
        return False
    
    def get_predicted_box(self):
        """Get predicted bounding box"""
        sx, sy = self.kf.x[0], self.kf.x[1]
        
        if self.last_detection is not None:
            x1, y1, x2, y2 = self.last_detection
            w, h = x2 - x1, y2 - y1
        else:
            w, h = 50, 50
        
        return (int(sx - w/2), int(sy - h/2), int(sx + w/2), int(sy + h/2))