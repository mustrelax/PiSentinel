"""
MultiObjectTracker v2: Stable swarm tracking
Key changes:
1. Track merging for overlapping detections
2. Detection suppression in swarm regions
3. Stricter new track creation
4. Ghost track prevention
"""
import math
import numpy as np
from collections import deque

from tracking.track import Track
from tracking.matcher import HungarianMatcher


class MultiObjectTracker:
    """
    Main MOT orchestrator with swarm-aware stability features.
    """
    
    # TRACK MERGE PARAMETERS
    MERGE_DISTANCE = 70           # Merge tracks closer than this (pixels)
    MERGE_MIN_AGE = 3             # Only merge tracks older than this
    
    # NEW TRACK CREATION PARAMETERS (STRICTER)
    MIN_CONFIDENCE_NEW_TRACK = 0.30   # Slightly lower for stationary detection
    SUPPRESS_NEW_IN_SWARM_RADIUS = 100  # Larger radius - less new tracks near existing
    MIN_FRAMES_BETWEEN_NEW_TRACKS = 3   # Don't create new track every frame
    
    # DETECTION STABILIZATION
    DETECTION_BUFFER_SIZE = 3     # Buffer detections over N frames
    
    def __init__(self):
        self.tracks = []
        self.next_track_id = 1
        self.matcher = HungarianMatcher(max_iou_distance=0.7)  # Stricter matching
        
        # Active target management
        self.active_track_id = None
        self.switch_cooldown_counter = 0
        
        # Detection history for stabilization
        self.detection_history = deque(maxlen=self.DETECTION_BUFFER_SIZE)
        
        # Frame counter
        self.frame_count = 0
        
        # NEW: Track creation throttle
        self.last_track_creation_frame = 0
    
    def _should_suppress_new_track(self, detection, confidences_idx):
        """
        Check if we should suppress creation of a new track here.
        Prevents ID explosion in swarm regions.
        """
        x1, y1, x2, y2 = detection
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Check if too close to any existing confirmed track
        for track in self.tracks:
            if track.state != Track.CONFIRMED:
                continue
            
            tx, ty = track.kf.x[0], track.kf.x[1]
            distance = math.sqrt((cx - tx)**2 + (cy - ty)**2)
            
            if distance < self.SUPPRESS_NEW_IN_SWARM_RADIUS:
                # Too close to existing track - suppress
                return True
        
        return False
    
    def _merge_overlapping_tracks(self):
        """
        Merge tracks that are too close together.
        Keeps the older/more established track.
        """
        if len(self.tracks) < 2:
            return
        
        merged_ids = set()
        
        for i, track_a in enumerate(self.tracks):
            if track_a.track_id in merged_ids:
                continue
            if track_a.state != Track.CONFIRMED:
                continue
            if track_a.age < self.MERGE_MIN_AGE:
                continue
            
            for j, track_b in enumerate(self.tracks):
                if i >= j:
                    continue
                if track_b.track_id in merged_ids:
                    continue
                if track_b.state != Track.CONFIRMED:
                    continue
                
                distance = track_a.distance_to(track_b)
                
                if distance < self.MERGE_DISTANCE:
                    # Merge: keep older track, delete newer
                    if track_a.age >= track_b.age:
                        keep_track = track_a
                        delete_track = track_b
                    else:
                        keep_track = track_b
                        delete_track = track_a
                    
                    # Transfer hits to keep track
                    keep_track.hits += delete_track.hits // 2
                    
                    merged_ids.add(delete_track.track_id)
                    print(f"[MERGE] Track {delete_track.track_id} → {keep_track.track_id} (dist:{distance:.0f}px)")
        
        # Remove merged tracks
        if merged_ids:
            self.tracks = [t for t in self.tracks if t.track_id not in merged_ids]
    
    def _cleanup_ghost_tracks(self):
        """
        Remove tracks that have truly diverged.
        Be CONSERVATIVE - don't delete tracks that might return!
        """
        ghost_ids = []
        
        for track in self.tracks:
            sx, sy = track.kf.x[0], track.kf.x[1]
            vx, vy = track.kf.x[2], track.kf.x[3]
            speed = math.sqrt(vx**2 + vy**2)
            
            # Only delete if REALLY diverged
            if speed > 120:  # Very unrealistic speed
                ghost_ids.append(track.track_id)
                continue
            
            # Only delete if WAY outside screen
            margin = 150
            if (sx < -margin or sx > config.FRAME_WIDTH + margin or
                sy < -margin or sy > config.FRAME_HEIGHT + margin):
                ghost_ids.append(track.track_id)
                continue
            
            # High uncertainty check - only for old lost tracks
            if track.state == track.LOST and track.time_since_update > 30:
                pos_uncertainty = track.kf.P[0, 0] + track.kf.P[1, 1]
                if pos_uncertainty > 50000:  # Very high uncertainty
                    ghost_ids.append(track.track_id)
                    continue
        
        if ghost_ids:
            print(f"[GHOST] Removing: {ghost_ids}")
            self.tracks = [t for t in self.tracks if t.track_id not in ghost_ids]
    
    def _filter_detections(self, detections, confidences):
        """
        Filter and stabilize detections.
        Returns (filtered_detections, filtered_confidences)
        """
        if len(detections) == 0:
            return [], []
        
        filtered_dets = []
        filtered_confs = []
        
        for det, conf in zip(detections, confidences):
            x1, y1, x2, y2 = det
            
            # Filter by size (too small = noise, too big = merged detection)
            w, h = x2 - x1, y2 - y1
            
            if w < 15 or h < 15:  # Too small
                continue
            if w > 300 or h > 300:  # Too big (probably merged)
                continue
            
            # Filter by aspect ratio (drones are roughly square-ish)
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 4.0:  # Too elongated
                continue
            
            # Filter by position (partially outside frame = uncertain)
            if x1 < -20 or y1 < -20:
                continue
            if x2 > config.FRAME_WIDTH + 20 or y2 > config.FRAME_HEIGHT + 20:
                continue
            
            filtered_dets.append(det)
            filtered_confs.append(conf)
        
        return filtered_dets, filtered_confs
    
    def update(self, detections, confidences):
        """
        Main update loop with swarm-aware stability features.
        """
        self.frame_count += 1
        
        # STEP 0: Filter detections
        detections, confidences = self._filter_detections(detections, confidences)
        
        # STEP 1: Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # STEP 2: Associate detections to tracks (Hungarian)
        matches, unmatched_tracks, unmatched_detections = self.matcher.match(
            self.tracks, detections
        )
        
        # STEP 3: Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], confidences[det_idx])
        
        # STEP 4: Mark missed tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # STEP 5: Create new tracks (with suppression and throttling)
        tracks_created_this_frame = 0
        max_new_per_frame = 2  # Don't create more than 2 new tracks per frame
        
        for det_idx in unmatched_detections:
            # Throttle: max N new tracks per frame
            if tracks_created_this_frame >= max_new_per_frame:
                break
            
            det = detections[det_idx]
            conf = confidences[det_idx]
            
            # Skip low confidence detections
            if conf < self.MIN_CONFIDENCE_NEW_TRACK:
                continue
            
            # Skip if too close to existing track (swarm region)
            if self._should_suppress_new_track(det, det_idx):
                continue
            
            # Throttle: Don't create tracks too frequently
            if (self.frame_count - self.last_track_creation_frame) < self.MIN_FRAMES_BETWEEN_NEW_TRACKS:
                # Allow only if we have NO confirmed tracks
                if any(t.state == Track.CONFIRMED for t in self.tracks):
                    continue
            
            # Create new track
            new_track = Track(self.next_track_id, det, conf)
            self.tracks.append(new_track)
            self.next_track_id += 1
            self.last_track_creation_frame = self.frame_count
            tracks_created_this_frame += 1
        
        # STEP 6: Merge overlapping tracks
        self._merge_overlapping_tracks()
        
        # STEP 7: Cleanup ghost tracks
        self._cleanup_ghost_tracks()
        
        # STEP 8: Delete old tracks
        before_count = len(self.tracks)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        deleted_count = before_count - len(self.tracks)
        
        if deleted_count > 0:
            print(f"[TRACK] Deleted {deleted_count} track(s), remaining: {len(self.tracks)}")
    
    def get_all_tracks(self):
        """Return all tracks for external processing"""
        return self.tracks
    
    def get_confirmed_tracks(self):
        """Return only confirmed tracks"""
        return [t for t in self.tracks if t.state == Track.CONFIRMED]
    
    def get_track_count(self):
        """Return count of confirmed tracks"""
        return len(self.get_confirmed_tracks())


# Need to import config for frame dimensions
import config