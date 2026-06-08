# io_devices.py
import cv2
import time
import socket
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from abc import ABC, abstractmethod

import logging
import config

# Setup logger
logger = logging.getLogger(__name__)

# --- Input Sources ---

class BaseInputSource(ABC):
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def read(self):
        # Returns (success_bool, frame_ndarray)
        pass

    @abstractmethod
    def cleanup(self):
        pass

class RealCameraInput(BaseInputSource):
    def __init__(self):
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            cam_cfg = self.picam2.create_video_configuration(
                main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT), "format": "RGB888"},
                raw={"size": self.picam2.sensor_resolution}
            )
            self.picam2.configure(cam_cfg)
            logger.info("Picamera2 configured.")
        except ImportError:
            logger.error("FATAL ERROR: Picamera2 library not found. Install it for 'real_camera' mode.")
            raise
        except Exception as e:
            logger.error(f"FATAL ERROR: Failed to initialize Picamera2: {e}")
            raise
            
    def start(self):
        self.picam2.start()
        logger.info("Picamera2 started.")
        time.sleep(1.0) # Allow camera to warm up
    
    def read(self):
        frame = self.picam2.capture_array()
        return True, frame

    def cleanup(self):
        try:
            self.picam2.stop()
            logger.info("Picamera2 stopped.")
        except Exception as e:
            logger.error(f"Error stopping Picamera2: {e}")

class SimulationInput(BaseInputSource):
    def __init__(self):
        CAPTURE_INDEX = config.SIM_CAPTURE_INDEX
        logger.info(f"Attempting to open capture card at index {CAPTURE_INDEX}...")
        self.video_capture = cv2.VideoCapture(CAPTURE_INDEX, cv2.CAP_V4L2)
        
        if not self.video_capture.isOpened():
            logger.error(f"FATAL ERROR: Capture card could not be opened at index {CAPTURE_INDEX}.")
            raise RuntimeError("Failed to open capture card")
        
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.video_capture.set(cv2.CAP_PROP_FPS, 60)
        
        width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        logger.info(f"Capture card OK. Settings: {int(width)}x{int(height)} @ {fps:.2f} FPS")

    def start(self):
        # cv2.VideoCapture is already "started" on init
        pass
    
    def read(self):
        ret, frame = self.video_capture.read()
        if not ret:
            logger.warning("Warning: Could not read frame from capture card.")
            return ret, frame

        if config.SIM_DUAL_MODE:
            h, w = frame.shape[:2]
            mid = w // 2
            if config.SIM_CAPTURE_SIDE == "right":
                frame = frame[:, mid:]
            else:
                frame = frame[:, :mid]

        if frame.shape[1] != config.FRAME_WIDTH or frame.shape[0] != config.FRAME_HEIGHT:
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                               interpolation=cv2.INTER_AREA)

        return ret, frame

    def cleanup(self):
        try:
            self.video_capture.release()
            logger.info("Capture card released.")
        except Exception as e:
            logger.error(f"Error releasing capture card: {e}")
            
# --- Output Controllers ---

class ServoController:
    def __init__(self):
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c, address=0x40)
            self.pca.frequency = 50
            self.pan_servo = servo.Servo(self.pca.channels[0])
            self.tilt_servo = servo.Servo(self.pca.channels[1])
            
            self.set_angles(config.PAN_START, config.TILT_START)
            logger.info("Servo controller initialized and centered.")
        except Exception as e:
            logger.error(f"FATAL ERROR: Could not initialize PCA9685 servos: {e}")
            logger.warning("Continuing without servo control (simulation only?).")
            self.pca = None
            self.pan_servo = None
            self.tilt_servo = None

    def set_angles(self, pan_angle, tilt_angle):
        if not self.pan_servo or not self.tilt_servo:
            return

        # Clamp values
        pan_safe = max(config.PAN_MIN, min(config.PAN_MAX, pan_angle))
        tilt_safe = max(config.TILT_MIN, min(config.TILT_MAX, tilt_angle))
        
        self.pan_servo.angle = pan_safe
        self.tilt_servo.angle = tilt_safe

    def cleanup(self):
        if not self.pca:
            return
        try:
            logger.info("Centering servos and releasing channels...")
            self.set_angles(config.PAN_START, config.TILT_START)
            time.sleep(0.5)
            self.pca.deinit()
        except Exception as e:
            logger.error(f"Error during servo cleanup: {e}")

class SimulationCommander:
    def __init__(self):
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_addr = (config.SIM_PC_IP, config.SIM_CMD_PORT)
        logger.info(f"Will send commands to Unreal at {self.target_addr}")

    def send_command(self, pan_angle, tilt_angle):
        # Convert servo angles (0-180) to simulation angles
        sim_pan = 90.0 - pan_angle
        sim_tilt = 90.0 - tilt_angle
        
        message = f"{sim_pan},{sim_tilt}"
        self.command_socket.sendto(message.encode(), self.target_addr)

    def cleanup(self):
        try:
            self.command_socket.close()
            logger.info("Simulation command socket closed.")
        except Exception as e:
            logger.error(f"Error closing command socket: {e}")
