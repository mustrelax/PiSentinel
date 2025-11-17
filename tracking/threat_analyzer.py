"""
ThreatAnalyzer: Computes threat scores and selects active target based on
distance, velocity, approach angle, and confidence.
"""
import math
import numpy as np

import config
from tracking.track import Track


class ThreatAnalyzer:
    """
    Analyzes tracks and computes threat scores for prioritization.
    Implements hysteresis-based target switching to prevent oscillation.
    """
    
    # Threat scoring weights
    WEIGHT_DISTANCE = 100.0
    WEIGHT_VELOCITY = 15.0
    WEIGHT_APPROACH = 50.0
    WEIGHT_CONFIDENCE = 5.0
    
    @staticmethod
    def calculate_threat_score(track):
        """
        Calculate threat score for a single track.
        Higher score = higher threat.
        """
        if track.state != Track.CONFIRMED:
            return 0.0
        
        sx, sy = track.kf.x[0], track.kf.x[1]
        vx, vy = track.kf.x[2], track.kf.x[3]
        
        # 1. DISTANCE SCORE (inverse - closer = higher threat)
        cx_screen = config.FRAME_WIDTH / 2
        cy_screen = config.FRAME_HEIGHT / 2
        distance = math.sqrt((sx - cx_screen)**2 + (sy - cy_screen)**2)
        max_distance = math.sqrt(cx_screen**2 + cy_screen**2)
        distance_score = ThreatAnalyzer.WEIGHT_DISTANCE * (1.0 - distance / max_distance)
        
        # 2. VELOCITY SCORE (faster = higher threat)
        velocity = math.sqrt(vx**2 + vy**2)
        velocity_score = ThreatAnalyzer.WEIGHT_VELOCITY * velocity
        
        # Bonus for aggressive motion
        if track.is_aggressive_motion():
            velocity_score *= 1.5
        
        # 3. APPROACH ANGLE SCORE (heading towards center = higher threat)
        to_center = np.array([cx_screen - sx, cy_screen - sy])
        velocity_vec = np.array([vx, vy])
        
        to_center_norm = np.linalg.norm(to_center)
        velocity_norm = np.linalg.norm(velocity_vec)
        
        if to_center_norm > 1e-6 and velocity_norm > 1e-6:
            cos_angle = np.dot(velocity_vec, to_center) / (velocity_norm * to_center_norm)
            approach_factor = max(0, cos_angle)
        else:
            approach_factor = 0.0
        
        approach_score = ThreatAnalyzer.WEIGHT_APPROACH * approach_factor
        
        # 4. CONFIDENCE SCORE (longer track = more reliable)
        confidence_score = ThreatAnalyzer.WEIGHT_CONFIDENCE * min(track.hits / 10.0, 1.0)
        
        total_score = distance_score + velocity_score + approach_score + confidence_score
        
        return total_score
    
    @staticmethod
    def select_active_target(tracks, current_active_id, switch_cooldown_counter):
        """
        Select the track with highest threat score as active target.
        Implements hysteresis to prevent frequent switching.
        
        Args:
            tracks: List of Track objects
            current_active_id: Currently tracked target ID (or None)
            switch_cooldown_counter: Frames since last switch
        
        Returns:
            (active_track, should_switch)
        """
        SWITCH_THRESHOLD = 1.3
        SWITCH_COOLDOWN = 10
        
        confirmed_tracks = [t for t in tracks if t.state == Track.CONFIRMED]
        
        if len(confirmed_tracks) == 0:
            return None, False
        
        # Calculate threat scores
        for track in confirmed_tracks:
            track.threat_score = ThreatAnalyzer.calculate_threat_score(track)
        
        confirmed_tracks.sort(key=lambda t: t.threat_score, reverse=True)
        best_track = confirmed_tracks[0]
        
        if current_active_id is None:
            return best_track, True
        
        current_track = next((t for t in confirmed_tracks if t.track_id == current_active_id), None)
        
        if current_track is None or current_track.state != Track.CONFIRMED:
            return best_track, True
        
        # Emergency switch for fast-moving threats
        if best_track.is_aggressive_motion() and best_track.track_id != current_active_id:
            print(f"[SWITCH] EMERGENCY: Fast threat #{best_track.track_id}")
            return best_track, True
        
        if switch_cooldown_counter < SWITCH_COOLDOWN:
            return current_track, False
        
        if best_track.threat_score > current_track.threat_score * SWITCH_THRESHOLD:
            print(f"[SWITCH] {current_active_id} -> {best_track.track_id} (Threat: {best_track.threat_score:.0f} > {current_track.threat_score:.0f})")
            return best_track, True
        
        return current_track, False