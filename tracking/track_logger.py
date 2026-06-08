"""
Track Logger Module - Logs track summaries on completion.
Outputs summary CSV (not per-frame) and JSON for dashboard.

DIRECTION CALCULATION:
    Uses config.CAMERA_FACING_DIRECTION as base direction.
    Screen right (+X) adds to bearing, screen left (-X) subtracts.
    Example: Camera facing West (270°), target moves right = North (0°)
"""
import os
import csv
import math
import logging
from datetime import datetime
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# Log storage directory
LOCAL_LOG_DIR = Path(__file__).parent.parent / "logs"

# Cardinal direction labels
DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


def bearing_to_direction(bearing: float) -> str:
    """Convert bearing (0-360°) to cardinal direction."""
    bearing = bearing % 360
    index = int((bearing + 22.5) / 45) % 8
    return DIRECTIONS[index]


def screen_to_world_bearing(screen_x: float, screen_y: float, 
                            center_x: float, center_y: float,
                            camera_facing: float) -> float:
    """
    Convert screen position to world compass bearing.
    
    Args:
        screen_x, screen_y: Position on screen
        center_x, center_y: Screen center
        camera_facing: Camera's compass bearing (0=N, 90=E, 180=S, 270=W)
    
    Returns:
        World bearing (0-360°) where target is located
    """
    # Vector from center to target
    dx = screen_x - center_x  # Positive = Right of center
    dy = screen_y - center_y  # Positive = Below center (screen coords)
    
    # Screen-relative angle: right=0°, up=-90°, left=180°, down=90°
    screen_angle = math.degrees(math.atan2(dy, dx))
    
    # Convert to world bearing
    # Screen right (+X) = camera's right = camera_facing + 90°
    # Screen up (-Y) = camera's forward = camera_facing
    world_bearing = camera_facing + screen_angle + 90
    
    # Normalize to 0-360
    world_bearing = world_bearing % 360
    
    return round(world_bearing, 1)


class TrackLogger:
    """Logs track summary on finalization (not per-frame)."""
    
    def __init__(self, track_id: int, sentinel_name: str):
        self.track_id = track_id
        self.sentinel_name = sentinel_name
        self.start_time = datetime.now()
        self.frame_count = 0
        
        # Track positions for direction calculation
        self.entry_pos = None
        self.exit_pos = None
        self.last_pos = None
        
        # Stats (using 2D pixel speed, not buggy 3D)
        self.speed_samples = []
        self.distance_samples = []
        
        # Track type (was it ever the locked/active target?)
        self.was_ever_locked = False
        
        # Screen center
        self.center_x = config.FRAME_WIDTH / 2
        self.center_y = config.FRAME_HEIGHT / 2
        
        logger.info(f"[TRACKLOG] Started Track #{track_id}")
    
    def update(self, track, frame_count: int, is_locked: bool = False):
        """Called every frame to collect data. Lightweight."""
        self.frame_count = frame_count
        
        # Track if this was ever the locked target
        if is_locked:
            self.was_ever_locked = True
        
        # Get position
        pos_x, pos_y = track.kf.x[0], track.kf.x[1]
        
        # Store entry position (first detection)
        if self.entry_pos is None:
            self.entry_pos = (pos_x, pos_y)
        
        # Always update last position
        self.last_pos = (pos_x, pos_y)
        self.exit_pos = (pos_x, pos_y)
        
        # Sample speed (2D pixel speed - more reliable)
        # Convert px/frame to approximate m/s using focal length ratio
        pixel_speed = track.get_current_speed()  # px/frame
        
        # Only record meaningful speeds (> 0.5 px/frame)
        if pixel_speed > 0.5:
            self.speed_samples.append(pixel_speed)
        
        # Sample distance if available
        if hasattr(track, 'real_pos') and track.real_pos[2] > 0.1:
            self.distance_samples.append(track.real_pos[2])
    
    def finalize(self, reason: str = "LOST") -> dict:
        """Generate summary and write to CSV."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Calculate world bearings
        camera_facing = config.CAMERA_FACING_DIRECTION
        
        entry_bearing = 0.0
        exit_bearing = 0.0
        
        if self.entry_pos:
            entry_bearing = screen_to_world_bearing(
                self.entry_pos[0], self.entry_pos[1],
                self.center_x, self.center_y, camera_facing
            )
        
        if self.exit_pos:
            exit_bearing = screen_to_world_bearing(
                self.exit_pos[0], self.exit_pos[1],
                self.center_x, self.center_y, camera_facing
            )
        
        # Calculate stats
        avg_speed_px = sum(self.speed_samples) / len(self.speed_samples) if self.speed_samples else 0
        max_speed_px = max(self.speed_samples) if self.speed_samples else 0
        
        avg_distance = sum(self.distance_samples) / len(self.distance_samples) if self.distance_samples else 0
        min_distance = min(self.distance_samples) if self.distance_samples else 0
        max_distance = max(self.distance_samples) if self.distance_samples else 0
        
        # False detection heuristics:
        # - Very short tracks
        # - No meaningful movement with sparse samples
        # - Passive short tracks in cluttered scenes (never became locked)
        possible_false = False
        if duration < 2.0:
            possible_false = True
        elif avg_speed_px < 0.5 and len(self.speed_samples) < 10:
            possible_false = True
        elif len(self.speed_samples) < 5:
            possible_false = True
        elif (not self.was_ever_locked) and duration < 4.0 and avg_speed_px < 1.2:
            possible_false = True
        
        # Build summary
        summary = {
            'track_id': self.track_id,
            'sentinel_name': self.sentinel_name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_sec': round(duration, 1),
            'total_frames': self.frame_count,
            'reason': reason,
            
            # Direction (world-relative)
            'entry_bearing': entry_bearing,
            'entry_direction': bearing_to_direction(entry_bearing),
            'exit_bearing': exit_bearing,
            'exit_direction': bearing_to_direction(exit_bearing),
            
            # Last known position (screen coords)
            'last_pos_x': round(self.last_pos[0], 1) if self.last_pos else 0,
            'last_pos_y': round(self.last_pos[1], 1) if self.last_pos else 0,
            
            # Speed (in px/frame - honest metric)
            'avg_speed_px': round(avg_speed_px, 2),
            'max_speed_px': round(max_speed_px, 2),
            
            # Distance
            'avg_distance_m': round(avg_distance, 2),
            'min_distance_m': round(min_distance, 2),
            'max_distance_m': round(max_distance, 2),
            
            # Camera info
            'camera_facing': camera_facing,
            
            # Track type: ACTIVE (was ever locked) or PASSIVE (never locked)
            'track_type': 'ACTIVE' if self.was_ever_locked else 'PASSIVE',
            
            # False detection flag
            'possible_false': possible_false
        }
        
        # Write summary to date-based CSV
        csv_path = self._write_csv(summary)
        if csv_path:
            summary['csv_path'] = str(csv_path)
        
        logger.info(
            f"[TRACKLOG] Track #{self.track_id}: {duration:.1f}s, "
            f"{bearing_to_direction(entry_bearing)}->{bearing_to_direction(exit_bearing)}"
        )
        
        return summary
    
    def _write_csv(self, summary: dict) -> Path | None:
        """Write summary to date-based CSV file (one row per track)."""
        LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        date_str = self.start_time.strftime("%Y-%m-%d")
        filename = f"tracks_{self.sentinel_name}_{date_str}.csv"
        filepath = LOCAL_LOG_DIR / filename
        
        # Define columns for summary CSV
        columns = [
            'track_id', 'track_type', 'possible_false', 'start_time', 'end_time', 'duration_sec',
            'entry_direction', 'exit_direction', 'entry_bearing', 'exit_bearing',
            'last_pos_x', 'last_pos_y', 'avg_speed_px', 'max_speed_px',
            'avg_distance_m', 'min_distance_m', 'max_distance_m', 'reason'
        ]
        
        file_exists = filepath.exists()
        
        try:
            with open(filepath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                if not file_exists:
                    writer.writeheader()
                writer.writerow(summary)
            return filepath
        except Exception as e:
            logger.error(f"[TRACKLOG] CSV write failed: {e}")
            return None


class TrackLoggerManager:
    """Manages loggers for all active tracks."""
    
    def __init__(self, sentinel_name: str, on_log_complete=None):
        self.sentinel_name = sentinel_name
        self.loggers = {}
        self.on_log_complete = on_log_complete
    
    def start_logging(self, track_id: int):
        if track_id not in self.loggers:
            self.loggers[track_id] = TrackLogger(track_id, self.sentinel_name)
    
    def update(self, track, frame_count: int, is_locked: bool = False):
        if track.track_id in self.loggers:
            self.loggers[track.track_id].update(track, frame_count, is_locked)
    
    def finalize(self, track_id: int, reason: str = "LOST"):
        if track_id in self.loggers:
            summary = self.loggers[track_id].finalize(reason)
            del self.loggers[track_id]
            
            if self.on_log_complete:
                try:
                    self.on_log_complete(summary)
                except Exception as e:
                    logger.error(f"[TRACKLOG] Callback failed: {e}")
            
            return summary
        return None
    
    def get_active_count(self) -> int:
        return len(self.loggers)
    
    def finalize_all(self, reason: str = "DISCONNECTED") -> list:
        """Finalize all active loggers (e.g., on disconnect)."""
        summaries = []
        for track_id in list(self.loggers.keys()):
            summary = self.finalize(track_id, reason)
            if summary:
                summaries.append(summary)
        return summaries
