# Video Logger Module
# Records video of tracked objects with HUD overlay.

import cv2
import logging
import requests
from pathlib import Path
from datetime import datetime

import config

logger = logging.getLogger("VideoLogger")

# Local temp video directory (will be uploaded to server)
VIDEO_LOG_DIR = Path("./vlogs")


class VideoLogger:
    """Records frames for a single track."""
    
    def __init__(self, track_id: int, sentinel_name: str, frame_size: tuple, fps: int = 30):
        self.track_id = track_id
        self.sentinel_name = sentinel_name
        self.frame_size = frame_size  # (width, height)
        self.fps = fps
        self.start_time = datetime.now()
        self.frame_count = 0
        self.writer = None
        self.filepath = None
        self.filename = None
        
        # Initialize video writer
        self._init_writer()
        
        logger.info(f"[VIDEOLOG] Started recording Track #{track_id}")
    
    def _init_writer(self):
        """Initialize OpenCV VideoWriter."""
        VIDEO_LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Filename: {sentinel_name}_track{id}_{timestamp}.mp4
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.filename = f"{self.sentinel_name}_track{self.track_id}_{timestamp}.mp4"
        self.filepath = VIDEO_LOG_DIR / self.filename
        
        # Use mp4v codec (widely compatible)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.writer = cv2.VideoWriter(
            str(self.filepath),
            fourcc,
            self.fps,
            self.frame_size
        )
        
        if not self.writer.isOpened():
            logger.error(f"[VIDEOLOG] Failed to open video writer for Track #{self.track_id}")
            self.writer = None
    
    def add_frame(self, frame):
        """Add a frame to the recording."""
        if self.writer is None:
            return
        
        # Ensure frame matches expected size
        h, w = frame.shape[:2]
        if (w, h) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def finalize(self) -> dict:
        """Finalize recording and return summary."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        summary = {
            'track_id': self.track_id,
            'video_filename': self.filename,
            'video_path': None,  # Will be set after upload
            'duration_sec': round(duration, 1),
            'frame_count': self.frame_count,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
        
        if self.filepath and self.filepath.exists():
            file_size_mb = self.filepath.stat().st_size / (1024 * 1024)
            summary['file_size_mb'] = round(file_size_mb, 2)
            logger.info(
                f"[VIDEOLOG] Track #{self.track_id}: {duration:.1f}s, "
                f"{self.frame_count} frames, {file_size_mb:.2f}MB -> {self.filename}"
            )
            
            # Upload to server
            uploaded = self._upload_to_server()
            if uploaded:
                summary['video_path'] = f"/vlogs/{self.filename}"
                # Delete local file after successful upload
                try:
                    self.filepath.unlink()
                except:
                    pass
        else:
            logger.warning(f"[VIDEOLOG] Track #{self.track_id}: No video file created")
        
        return summary
    
    def _upload_to_server(self) -> bool:
        """Upload video file to central server."""
        if not self.filepath or not self.filepath.exists():
            return False
        
        try:
            server_url = config.CENTRAL_SERVER_URL
            upload_url = f"{server_url}/upload-video"
            
            with open(self.filepath, 'rb') as f:
                files = {'video': (self.filename, f, 'video/mp4')}
                response = requests.post(upload_url, files=files, timeout=60)
            
            if response.status_code == 200:
                logger.info(f"[VIDEOLOG] Uploaded {self.filename} to server")
                return True
            else:
                logger.error(f"[VIDEOLOG] Upload failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"[VIDEOLOG] Upload error: {e}")
            return False


class VideoLoggerManager:
    """Manages video recordings for tracks."""
    
    def __init__(self, sentinel_name: str, frame_size: tuple = None, fps: int = 30):
        self.sentinel_name = sentinel_name
        self.frame_size = frame_size or (config.FRAME_WIDTH, config.FRAME_HEIGHT)
        self.fps = fps
        self.loggers = {}  # track_id -> VideoLogger
        self.current_locked_id = None  # Only record the locked track
    
    def start_logging(self, track_id: int):
        """Start recording for a track (called when track becomes LOCKED)."""
        if track_id in self.loggers:
            return  # Already recording
        
        self.loggers[track_id] = VideoLogger(
            track_id,
            self.sentinel_name,
            self.frame_size,
            self.fps
        )
        self.current_locked_id = track_id
    
    def add_frame(self, track_id: int, frame):
        """Add frame to track's recording (only if it's the locked track)."""
        if track_id in self.loggers:
            self.loggers[track_id].add_frame(frame)
    
    def add_frame_to_locked(self, frame):
        """Add frame to currently locked track's recording."""
        if self.current_locked_id and self.current_locked_id in self.loggers:
            self.loggers[self.current_locked_id].add_frame(frame)
    
    def set_locked_track(self, track_id: int):
        """Update which track is currently locked (for TTI switches)."""
        if track_id != self.current_locked_id:
            # Start recording new locked track if not already
            if track_id not in self.loggers:
                self.start_logging(track_id)
            self.current_locked_id = track_id
    
    def finalize(self, track_id: int) -> dict:
        """Finalize recording for a track."""
        if track_id in self.loggers:
            summary = self.loggers[track_id].finalize()
            del self.loggers[track_id]
            
            if self.current_locked_id == track_id:
                self.current_locked_id = None
            
            return summary
        return None
    
    def finalize_all(self) -> list:
        """Finalize all active recordings (e.g., on disconnect)."""
        summaries = []
        for track_id in list(self.loggers.keys()):
            summary = self.finalize(track_id)
            if summary:
                summaries.append(summary)
        return summaries
    
    def get_active_count(self) -> int:
        """Return number of active recordings."""
        return len(self.loggers)
