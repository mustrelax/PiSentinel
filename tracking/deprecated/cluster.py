"""
DroneCluster: Cluster representation with Kalman filtering.
Identical structure to Track.py from SOT system.
"""
import math
import numpy as np
from filterpy.kalman import KalmanFilter
from collections import deque

import config


class DroneCluster:
    """
    Represents a cluster of drones with Kalman filter state estimation.
    Motion-adaptive prediction just like individual Track.
    
    State vector: [cx, cy, vx, vy]
    - cx, cy: Cluster center position
    - vx, vy: Cluster center velocity
    """
    
    # Cluster states (identical to Track)
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    DELETED = 3
    
    # Speed thresholds (pixels/frame) - same as Track
    FAST_MOTION_THRESHOLD = 10.0      # Fast motion
    AGGRESSIVE_MODE_THRESHOLD = 15.0  # Aggressive motion
    
    def __init__(self, cluster_id, tracks):
        self.cluster_id = cluster_id
        self.tracks = tracks
        self.size = len(tracks)
        
        # State management (identical to Track)
        self.state = self.TENTATIVE
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        
        # Initialize Kalman Filter (4D state: cx, cy, vx, vy)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # Calculate initial center from tracks
        cx, cy = self._calculate_center_from_tracks()
        
        # State vector: [cx, cy, vx, vy]
        self.kf.x = np.array([cx, cy, 0.0, 0.0])
        
        # State transition matrix (constant velocity model)
        # x(k) = F * x(k-1) + w(k)
        # Position: cx(k) = cx(k-1) + vx(k-1) * dt
        # Velocity: vx(k) = vx(k-1)
        self.kf.F = np.array([
            [1, 0, 1, 0],  # cx = cx + vx
            [0, 1, 0, 1],  # cy = cy + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=float)
        
        # Measurement matrix (we only measure position, not velocity)
        # z(k) = H * x(k) + v(k)
        self.kf.H = np.array([
            [1, 0, 0, 0],  # Measure cx
            [0, 1, 0, 0]   # Measure cy
        ], dtype=float)
        
        # Initial covariance matrix (high uncertainty)
        self.kf.P *= 500.0
        
        # Measurement noise covariance (R matrix) - INCREASED for more smoothing
        self.kf.R = np.array([
            [15, 0],  # Increased from 8 → less trust in jittery measurements
            [0, 15]
        ], dtype=float)
        
        # Process noise covariance (Q matrix) - DECREASED for more smoothing
        self.kf.Q = np.array([
            [8,  0,  0,  0],  # Decreased from 15 → smoother position
            [0,  8,  0,  0],
            [0,  0, 25,  0],  # Decreased from 40 → smoother velocity
            [0,  0,  0, 25]
        ], dtype=float)
        
        # Base Q for adaptive adjustment
        self.base_Q = self.kf.Q.copy()
        
        # DEEP TEMPORAL BUFFER (for ultra-smooth center)
        self.raw_center_history = deque(maxlen=10)  # Last 10 raw measurements
        self.raw_center_history.append((cx, cy))
        
        # Speed tracking (for motion classification)
        self.speed_history = deque(maxlen=5)
        self.speed_history.append(0.0)
        
        # Trajectory history
        self.trajectory = deque(maxlen=30)
        self.trajectory.append((cx, cy))
        
        # Bounding box
        self.bounding_box = self._calculate_bounding_box()
        
        # Threat score (calculated externally)
        self.threat_score = 0.0
    
    def _calculate_center_from_tracks(self):
        """Calculate geometric center from all tracks in cluster"""
        if not self.tracks:
            return (config.FRAME_WIDTH / 2, config.FRAME_HEIGHT / 2)
        
        xs = [t.kf.x[0] for t in self.tracks]
        ys = [t.kf.x[1] for t in self.tracks]
        
        return (np.mean(xs), np.mean(ys))
    
    def _calculate_bounding_box(self):
        """Calculate bounding box encompassing all tracks"""
        if not self.tracks:
            cx, cy = self.kf.x[0], self.kf.x[1]
            return (int(cx - 25), int(cy - 25), int(cx + 25), int(cy + 25))
        
        all_boxes = [t.get_predicted_box() for t in self.tracks]
        
        x1 = min([box[0] for box in all_boxes])
        y1 = min([box[1] for box in all_boxes])
        x2 = max([box[2] for box in all_boxes])
        y2 = max([box[3] for box in all_boxes])
        
        return (x1, y1, x2, y2)
    
    def get_current_speed(self):
        """Calculate current speed magnitude from velocity components"""
        vx, vy = self.kf.x[2], self.kf.x[3]
        return math.sqrt(vx**2 + vy**2)
    
    def get_average_speed(self):
        """Get smoothed speed over last 5 frames"""
        if not self.speed_history:
            return 0.0
        return np.mean(self.speed_history)
    
    def is_fast_moving(self):
        """Check if cluster is in fast motion"""
        return self.get_average_speed() > self.FAST_MOTION_THRESHOLD
    
    def is_aggressive_motion(self):
        """Check if cluster is in very aggressive motion"""
        return self.get_average_speed() > self.AGGRESSIVE_MODE_THRESHOLD
    
    def predict(self):
        """
        Kalman prediction with motion-adaptive Q matrix.
        Identical to Track.predict() logic.
        """
        # Update speed history
        current_speed = self.get_current_speed()
        self.speed_history.append(current_speed)
        
        # ADAPTIVE Q MATRIX (Motion-aware) - HEAVILY SMOOTHED
        # Stationary clusters should be ULTRA smooth
        if self.is_aggressive_motion():
            self.kf.Q = self.base_Q * 2.0  # Reduced from 2.5
            if self.age % 30 == 0:  # Log only occasionally
                print(f"[CLUSTER] C{self.cluster_id} AGGRESSIVE (speed: {current_speed:.1f})")
        elif self.is_fast_moving():
            self.kf.Q = self.base_Q * 1.5  # Reduced from 1.8
        else:
            # STATIONARY/SLOW → VERY LOW Q (maximum smoothness)
            self.kf.Q = self.base_Q * 0.3  # Much lower than 0.6
        
        # Kalman prediction step
        self.kf.predict()
        
        # Update age and time since last update
        self.age += 1
        self.time_since_update += 1
    
    def update(self, new_tracks):
        """
        Update cluster with new track associations.
        Performs Kalman update step with HEAVY temporal smoothing.
        """
        self.tracks = new_tracks
        self.size = len(new_tracks)
        
        # Calculate new RAW measured center
        raw_cx, raw_cy = self._calculate_center_from_tracks()
        
        # Add to history buffer
        self.raw_center_history.append((raw_cx, raw_cy))
        
        # WEIGHTED MOVING AVERAGE (smooth out detection jitter)
        # Use last 5-10 measurements with exponentially decreasing weights
        n = min(len(self.raw_center_history), 7)
        recent_centers = list(self.raw_center_history)[-n:]
        
        # Weights: newest gets most weight, but not 100%
        # [0.05, 0.08, 0.12, 0.18, 0.22, 0.25, 0.10] (sums to 1.0)
        if n == 7:
            weights = [0.05, 0.08, 0.12, 0.18, 0.22, 0.25, 0.10]
        elif n == 5:
            weights = [0.10, 0.15, 0.20, 0.25, 0.30]
        elif n == 3:
            weights = [0.20, 0.30, 0.50]
        else:
            weights = [1.0 / n] * n
        
        smoothed_cx = sum(c[0] * w for c, w in zip(recent_centers, weights))
        smoothed_cy = sum(c[1] * w for c, w in zip(recent_centers, weights))
        
        # ADAPTIVE R MATRIX (based on cluster size)
        adaptive_R = self.kf.R.copy()
        
        if self.size >= 3:
            adaptive_R *= 0.8  # Large cluster still somewhat stable
        elif self.size == 2:
            adaptive_R *= 1.2  # 2 drones less stable
        else:
            adaptive_R *= 1.5  # Single drone least stable
        
        # DETECTION CONFIDENCE WEIGHTING
        if self.tracks:
            avg_confidence = np.mean([t.last_confidence for t in self.tracks if hasattr(t, 'last_confidence')])
            confidence_scale = max(0.5, 1.5 - avg_confidence)  # Adjusted range
            adaptive_R *= confidence_scale
        
        # Kalman update with SMOOTHED measurement
        self.kf.update(z=np.array([smoothed_cx, smoothed_cy]), R=adaptive_R)
        
        # Update metadata
        self.hits += 1
        self.time_since_update = 0
        
        # Update trajectory
        sx, sy = self.kf.x[0], self.kf.x[1]
        self.trajectory.append((sx, sy))
        
        # Update bounding box
        self.bounding_box = self._calculate_bounding_box()
        
        # STATE TRANSITIONS (identical to Track logic)
        if self.state == self.TENTATIVE:
            # Faster confirmation for clusters (more reliable than single track)
            required_hits = 2 if self.size >= 2 else 3
            
            if self.hits >= required_hits:
                self.state = self.CONFIRMED
                print(f"[CLUSTER] C{self.cluster_id} CONFIRMED (size:{self.size}, hits:{self.hits})")
        
        elif self.state == self.LOST:
            self.state = self.CONFIRMED
            print(f"[CLUSTER] C{self.cluster_id} RE-ACQUIRED (lost:{self.time_since_update}f, size:{self.size})")
    
    def mark_missed(self):
        """Called when no detection matched this cluster"""
        if self.state == self.CONFIRMED:
            # Longer tolerance for fast-moving clusters
            lost_threshold = 20 if self.is_fast_moving() else 12
            
            if self.time_since_update > lost_threshold:
                self.state = self.LOST
                print(f"[CLUSTER] C{self.cluster_id} LOST (speed:{self.get_average_speed():.1f}, size:{self.size})")
    
    def is_deleted(self):
        """Check if cluster should be deleted"""
        # Delete tentative clusters quickly
        if self.state == self.TENTATIVE and self.time_since_update > 10:
            return True
        
        # Keep lost clusters longer if they were fast-moving
        if self.state == self.LOST:
            if self.is_fast_moving():
                return self.time_since_update > 90
            else:
                return self.time_since_update > 60
        
        return False
    
    def get_predicted_center(self):
        """Get Kalman-predicted center position"""
        return (self.kf.x[0], self.kf.x[1])
    
    def get_predicted_box(self):
        """Get predicted bounding box (for matching)"""
        # Use predicted center with last known size
        cx, cy = self.kf.x[0], self.kf.x[1]
        
        if self.bounding_box:
            x1, y1, x2, y2 = self.bounding_box
            w, h = x2 - x1, y2 - y1
        else:
            w, h = 100, 100  # Default size
        
        return (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))