# tracking_thread.py
import threading
import time
import datetime
import cv2

from webrtc_utils import FrameSource
from tracker import Tracker
from io_devices import BaseInputSource, ServoController, SimulationCommander

class TrackingThread(threading.Thread):
    
    FPS_BUFFER_SIZE = 30

    def __init__(self, 
                 frame_source: FrameSource,
                 input_device: BaseInputSource,
                 servo_controller: ServoController,
                 tracker: Tracker,
                 stop_event: threading.Event,
                 sim_commander: SimulationCommander | None = None):
        
        super().__init__(daemon=True, name="TrackingThread")
        self.frame_source = frame_source
        self.input_device = input_device
        self.servo_controller = servo_controller
        self.tracker = tracker
        self.stop_event = stop_event
        self.sim_commander = sim_commander
        
        self.fps_buffer = []

    def _add_timing_overlays(self, frame, t0):
        """Adds FPS and timestamp overlays to the frame."""
        # FPS overlay
        t1 = time.perf_counter()
        inst_fps = 1.0 / max(1e-6, (t1 - t0))
        self.fps_buffer.append(inst_fps)
        if len(self.fps_buffer) > self.FPS_BUFFER_SIZE:
            self.fps_buffer.pop(0)
        avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
        cv2.putText(frame, f"Avg FPS: {avg_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Timestamp
        timestamp_str = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        cv2.putText(frame, timestamp_str, (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def run(self):
        """Main loop: Read -> Process -> Control -> Publish"""
        print("Tracking thread started.")
        try:
            while not self.stop_event.is_set():
                t0 = time.perf_counter()

                # 1. Read from input device
                ret, frame = self.input_device.read()
                if not ret or frame is None:
                    time.sleep(0.01) # Avoid busy-looping if read fails
                    continue
                
                # 2. Process frame with tracker
                annotated_frame, pan, tilt = self.tracker.process_frame(frame)

                # 3. Control outputs
                self.servo_controller.set_angles(pan, tilt)
                if self.sim_commander:
                    self.sim_commander.send_command(pan, tilt)

                # 4. Add time-based overlays
                annotated_frame = self._add_timing_overlays(annotated_frame, t0)
                
                # 5. Publish frame to web server
                self.frame_source.set(annotated_frame)
        
        except Exception as e:
            print(f"FATAL ERROR in tracking thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # --- Cleanup ---
            print("Tracking thread stopping. Cleaning up resources...")
            self.input_device.cleanup()
            self.servo_controller.cleanup()
            if self.sim_commander:
                self.sim_commander.cleanup()
            cv2.destroyAllWindows()
            print("Tracking thread terminated.")