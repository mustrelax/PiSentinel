import threading
import time
import datetime
import cv2

import logging
from webrtc_utils import FrameSource
from tracking import TrackingController
from io_devices import BaseInputSource, ServoController, SimulationCommander

# Setup logger
logger = logging.getLogger(__name__)

class TrackingThread(threading.Thread):

    def __init__(self, 
                 frame_source: FrameSource,
                 input_device: BaseInputSource,
                 servo_controller: ServoController,
                 tracker: TrackingController,
                 stop_event: threading.Event,
                 sim_commander: SimulationCommander | None = None):
        
        super().__init__(daemon=True, name="TrackingThread")
        self.frame_source = frame_source
        self.input_device = input_device
        self.servo_controller = servo_controller
        self.tracker = tracker
        self.stop_event = stop_event
        self.sim_commander = sim_commander
        
    def run(self):
        logger.info("Tracking thread started.")
        self.last_loop_time = time.perf_counter()
        try:
            while not self.stop_event.is_set():

                # 1. Read from input device
                ret, frame = self.input_device.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Calculate True FPS (Cycle Time)
                t_now = time.perf_counter()
                dt = t_now - self.last_loop_time
                self.last_loop_time = t_now
                
                current_fps = 1.0 / max(1e-6, dt)

                annotated_frame, pan, tilt = self.tracker.process_frame(frame)

                self.servo_controller.set_angles(pan, tilt)
                if self.sim_commander:
                    self.sim_commander.send_command(pan, tilt)

                self.frame_source.set(annotated_frame)
        
        except Exception as e:
            logger.error(f"FATAL ERROR in tracking thread: {e}", exc_info=True)
        finally:
            # --- Cleanup ---
            logger.info("Tracking thread stopping. Cleaning up resources...")
            self.input_device.cleanup()
            self.servo_controller.cleanup()
            if self.sim_commander:
                self.sim_commander.cleanup()
            cv2.destroyAllWindows()
            logger.info("Tracking thread terminated.")