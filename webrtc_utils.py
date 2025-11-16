# webrtc_utils.py
from typing import Optional
from fractions import Fraction
import threading
import asyncio
import numpy as np
import av
from aiortc import VideoStreamTrack
import config

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

    def __init__(self, frame_source: FrameSource, target_fps: int):
        super().__init__()
        self.frame_source = frame_source
        self.fps = max(1, target_fps)
        self.frame_interval = 1.0 / self.fps
        self._ts = 0  # simple PTS counter

    async def recv(self) -> av.VideoFrame:
        # regulate FPS
        await asyncio.sleep(self.frame_interval)

        frame = self.frame_source.get()
        if frame is None:
            # Send a black frame if nothing is available yet
            frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)

        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        self._ts += 1
        video_frame.pts = self._ts
        video_frame.time_base = Fraction(1, self.fps)
        return video_frame