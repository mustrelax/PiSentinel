"""
SwarmAwareThreatAnalyzer v3: PRIMARY LOCK MODE
Key changes:
1. FIX: Same ID switch bug (was switching #1 → #1)
2. PRIMARY LOCK: Once locked, stay locked until target lost
3. IGNORE nearby drones in close-range swarm scenarios
4. Only switch on CRITICAL events (target lost, target escaped screen)
"""
import math
import numpy as np

import config
from tracking.track import Track


class SwarmAwareThreatAnalyzer:
    """
    PRIMARY LOCK threat analyzer.
    Once a target is selected, it stays locked until physically lost.
    Other drones are tracked visually but IGNORED for servo control.
    """
    
    # Base threat scoring weights
    WEIGHT_DISTANCE = 100.0
    WEIGHT_VELOCITY = 15.0
    WEIGHT_APPROACH = 50.0
    WEIGHT_CONFIDENCE = 5.0
    
    # SWARM AWARENESS (for visualization only now)
    SWARM_DETECTION_RADIUS = 200  # pixels
    
    # PRIMARY LOCK PARAMETERS
    LOCK_BONUS = 1000.0          # MASSIVE bonus - almost impossible to switch
    INITIAL_LOCK_FRAMES = 60     # 2 seconds of forced lock after selection
    
    # SWITCH CONDITIONS (very strict)
    SWITCH_COOLDOWN = 60         # 2 seconds minimum between switches
    EMERGENCY_ONLY = True        # Only switch on emergency (lost/escaped)
    
    # CLOSE RANGE DETECTION
    CLOSE_RANGE_BOX_SIZE = 120   # If bbox > this, drone is close
    
    @staticmethod
    def count_nearby_drones(track, all_tracks, radius):
        """Count nearby drones (for visualization/info only)"""
        tx, ty = track.kf.x[0], track.kf.x[1]
        
        nearby_count = 0
        for other_track in all_tracks:
            if other_track.track_id == track.track_id:
                continue
            if other_track.state != Track.CONFIRMED:
                continue
            
            ox, oy = other_track.kf.x[0], other_track.kf.x[1]
            distance = math.sqrt((tx - ox)**2 + (ty - oy)**2)
            
            if distance < radius:
                nearby_count += 1
        
        return nearby_count
    
    @staticmethod
    def is_close_range(track):
        """Check if drone is at close range (large bounding box)"""
        if track.last_detection is None:
            return False
        
        x1, y1, x2, y2 = track.last_detection
        w, h = x2 - x1, y2 - y1
        
        return max(w, h) > SwarmAwareThreatAnalyzer.CLOSE_RANGE_BOX_SIZE
    
    @staticmethod
    def is_on_screen(track, margin=50):
        """Check if track is within screen bounds"""
        sx, sy = track.kf.x[0], track.kf.x[1]
        
        return (sx >= -margin and sx <= config.FRAME_WIDTH + margin and
                sy >= -margin and sy <= config.FRAME_HEIGHT + margin)
    
    @staticmethod
    def calculate_threat_score(track, all_tracks, is_current_target=False):
        """
        Calculate threat score.
        Current target gets MASSIVE lock bonus.
        """
        if track.state != Track.CONFIRMED:
            return 0.0
        
        sx, sy = track.kf.x[0], track.kf.x[1]
        vx, vy = track.kf.x[2], track.kf.x[3]
        
        # ====== BASE THREAT ======
        
        # 1. DISTANCE SCORE
        cx_screen = config.FRAME_WIDTH / 2
        cy_screen = config.FRAME_HEIGHT / 2
        distance = math.sqrt((sx - cx_screen)**2 + (sy - cy_screen)**2)
        max_distance = math.sqrt(cx_screen**2 + cy_screen**2)
        distance_score = SwarmAwareThreatAnalyzer.WEIGHT_DISTANCE * (1.0 - distance / max_distance)
        
        # 2. VELOCITY SCORE
        velocity = math.sqrt(vx**2 + vy**2)
        velocity_score = SwarmAwareThreatAnalyzer.WEIGHT_VELOCITY * velocity
        
        if track.is_aggressive_motion():
            velocity_score *= 1.5
        
        # 3. APPROACH ANGLE SCORE
        to_center = np.array([cx_screen - sx, cy_screen - sy])
        velocity_vec = np.array([vx, vy])
        
        to_center_norm = np.linalg.norm(to_center)
        velocity_norm = np.linalg.norm(velocity_vec)
        
        if to_center_norm > 1e-6 and velocity_norm > 1e-6:
            cos_angle = np.dot(velocity_vec, to_center) / (velocity_norm * to_center_norm)
            approach_factor = max(0, cos_angle)
        else:
            approach_factor = 0.0
        
        approach_score = SwarmAwareThreatAnalyzer.WEIGHT_APPROACH * approach_factor
        
        # 4. CONFIDENCE SCORE
        confidence_score = SwarmAwareThreatAnalyzer.WEIGHT_CONFIDENCE * min(track.hits / 10.0, 1.0)
        
        # ====== PRIMARY LOCK BONUS ======
        lock_bonus = SwarmAwareThreatAnalyzer.LOCK_BONUS if is_current_target else 0.0
        
        # TOTAL
        total_score = (distance_score + velocity_score + approach_score + 
                      confidence_score + lock_bonus)
        
        # Store nearby count for visualization
        track.nearby_drone_count = SwarmAwareThreatAnalyzer.count_nearby_drones(
            track, all_tracks, SwarmAwareThreatAnalyzer.SWARM_DETECTION_RADIUS
        )
        
        return total_score
    
    @staticmethod
    def select_active_target(tracks, current_active_id, switch_cooldown_counter):
        """
        PRIMARY LOCK target selection with LOST track support.
        Key insight from 16-17/11: LOST tracks continue on Kalman prediction!
        """
        # Include CONFIRMED and LOST tracks (LOST still has valid Kalman state)
        valid_tracks = [t for t in tracks if t.state in [Track.CONFIRMED, Track.LOST]]
        
        # Prefer CONFIRMED, but LOST is acceptable
        confirmed_tracks = [t for t in valid_tracks if t.state == Track.CONFIRMED]
        
        if len(valid_tracks) == 0:
            return None, False
        
        # Calculate threat scores for all valid tracks
        for track in valid_tracks:
            is_current = (track.track_id == current_active_id)
            track.threat_score = SwarmAwareThreatAnalyzer.calculate_threat_score(
                track, valid_tracks, is_current_target=is_current
            )
        
        # Sort by threat
        valid_tracks.sort(key=lambda t: t.threat_score, reverse=True)
        best_track = valid_tracks[0]
        
        # Prefer confirmed track for new selection
        if confirmed_tracks:
            confirmed_tracks.sort(key=lambda t: t.threat_score, reverse=True)
            best_confirmed = confirmed_tracks[0]
        else:
            best_confirmed = best_track
        
        # ============ INITIAL TARGET SELECTION ============
        if current_active_id is None:
            target = best_confirmed
            close_str = " [CLOSE]" if SwarmAwareThreatAnalyzer.is_close_range(target) else ""
            print(f"[TARGET] PRIMARY LOCK → #{target.track_id}{close_str} (threat:{target.threat_score:.0f})")
            return target, True
        
        # ============ FIND CURRENT TARGET ============
        current_track = next((t for t in valid_tracks if t.track_id == current_active_id), None)
        
        # ============ EMERGENCY: Target Completely Gone ============
        if current_track is None:
            print(f"[TARGET] LOCK BROKEN: #{current_active_id} deleted → #{best_confirmed.track_id}")
            return best_confirmed, True
        
        # ============ Target is LOST but still has valid Kalman ============
        if current_track.state == Track.LOST:
            # Check if target was stationary (low speed history)
            avg_speed = current_track.get_average_speed()
            
            if avg_speed < 5.0:
                # STATIONARY TARGET: Wait much longer (3 sec)
                timeout = 90
            else:
                # MOVING TARGET: Standard timeout (1.5 sec)
                timeout = 45
            
            if current_track.time_since_update > timeout:
                print(f"[TARGET] LOCK TIMEOUT: #{current_active_id} lost {current_track.time_since_update}f → #{best_confirmed.track_id}")
                return best_confirmed, True
            
            # Otherwise keep tracking on prediction
            return current_track, False
        
        # ============ EMERGENCY: Target Escaped Screen ============
        if not SwarmAwareThreatAnalyzer.is_on_screen(current_track, margin=50):
            print(f"[TARGET] LOCK BROKEN: #{current_active_id} off-screen → #{best_confirmed.track_id}")
            return best_confirmed, True
        
        # ============ MAINTAIN LOCK ============
        return current_track, False
    
    @staticmethod
    def get_lock_status(current_track, all_tracks):
        """
        Get detailed lock status for HUD display.
        Returns dict with lock info.
        """
        if current_track is None:
            return {
                'locked': False,
                'target_id': None,
                'nearby_count': 0,
                'is_close_range': False,
                'ignored_count': 0
            }
        
        nearby = current_track.nearby_drone_count if hasattr(current_track, 'nearby_drone_count') else 0
        
        # Count how many tracks we're ignoring
        ignored = len([t for t in all_tracks 
                      if t.state == Track.CONFIRMED and t.track_id != current_track.track_id])
        
        return {
            'locked': True,
            'target_id': current_track.track_id,
            'nearby_count': nearby,
            'is_close_range': SwarmAwareThreatAnalyzer.is_close_range(current_track),
            'ignored_count': ignored
        }