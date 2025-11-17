"""
MultiObjectTracker: Orchestrates track lifecycle, performs data association,
and manages active target selection.
"""
from tracking.track import Track
from tracking.matcher import HungarianMatcher
from tracking.threat_analyzer import ThreatAnalyzer


class MultiObjectTracker:
    """
    Main MOT orchestrator. Manages all tracks and delegates to specialized classes.
    """
    
    def __init__(self):
        self.tracks = []
        self.next_track_id = 1
        self.matcher = HungarianMatcher(max_iou_distance=0.85)
        
        # Active target management
        self.active_track_id = None
        self.switch_cooldown_counter = 0
        
    def update(self, detections, confidences):
        """
        Main update loop: predict → associate → update → create → delete → select
        
        Args:
            detections: List of [x1, y1, x2, y2] boxes
            confidences: List of confidence scores
        
        Returns:
            active_track: The track being actively followed (or None)
        """
        # 1. PREDICT all tracks
        for track in self.tracks:
            track.predict()
        
        # 2. ASSOCIATE detections to tracks
        matches, unmatched_tracks, unmatched_detections = self.matcher.match(
            self.tracks, detections
        )
        
        # 3. UPDATE matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], confidences[det_idx])
        
        # 4. MARK MISSED tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # 5. CREATE new tracks
        for det_idx in unmatched_detections:
            new_track = Track(self.next_track_id, detections[det_idx], confidences[det_idx])
            self.tracks.append(new_track)
            print(f"[TRACK] New ID {self.next_track_id}")
            self.next_track_id += 1
        
        # 6. DELETE old tracks
        before_count = len(self.tracks)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        if len(self.tracks) < before_count:
            deleted_count = before_count - len(self.tracks)
            print(f"[TRACK] Deleted {deleted_count} track(s)")
        
        # 7. SELECT active target
        self.switch_cooldown_counter += 1
        active_track, switched = ThreatAnalyzer.select_active_target(
            self.tracks, self.active_track_id, self.switch_cooldown_counter
        )
        
        if switched:
            self.active_track_id = active_track.track_id if active_track else None
            self.switch_cooldown_counter = 0
        
        return active_track
    
    def get_all_tracks(self):
        """Return all tracks for visualization"""
        return self.tracks
    
    def get_active_track(self):
        """Return currently active track"""
        if self.active_track_id is None:
            return None
        return next((t for t in self.tracks if t.track_id == self.active_track_id), None)