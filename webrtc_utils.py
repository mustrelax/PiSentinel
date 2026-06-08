from typing import Optional
from fractions import Fraction
import threading
import asyncio
import logging
import numpy as np
import av
from aiortc import VideoStreamTrack
import config

logger = logging.getLogger(__name__)

class FrameSource:

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None

    def set(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

class WebRTCVideoTrack(VideoStreamTrack):

    kind = "video"
    CLOCK_RATE = 90000

    def __init__(self, frame_source: FrameSource, target_fps: int):
        super().__init__()
        self.frame_source = frame_source
        self.fps = max(1, target_fps)
        self._pts = 0
        self._pts_step = self.CLOCK_RATE // self.fps
        self._last_frame = None

    async def recv(self) -> av.VideoFrame:
        # Minimal sleep - just yield to event loop
        await asyncio.sleep(0.001)

        frame = self.frame_source.get()
        
        # Use last frame if no new one (avoid black frames)
        if frame is None:
            if self._last_frame is not None:
                frame = self._last_frame
            else:
                frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        else:
            self._last_frame = frame

        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = self._pts
        video_frame.time_base = Fraction(1, self.CLOCK_RATE)
        self._pts += self._pts_step
        return video_frame