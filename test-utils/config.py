# ----- Video / camera -----
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# ----- Model & tracking -----
#MODEL_PATH = "model_ncnn_model320"
MODEL_PATH = "yolov11n-UAV-finetune_ncnn_model320" 
TRACKER_CONFIG = "bytetrack.yaml"
FOCAL_LENGTH = 437.5
KNOWN_WIDTH_CM = 16.0
DEAD_ZONE_RADIUS = 20

# ----- PID -----
#PID_PAN = dict(Kp=0.06, Ki=0.01, Kd=0.005)
#PID_TILT = dict(Kp=0.07, Ki=0.01, Kd=0.005)
PID_PAN = dict(Kp=0.02, Ki=0.00, Kd=0.000)
PID_TILT = dict(Kp=0.02, Ki=0.00, Kd=0.000)
# ----- Servo limits (degrees) -----
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 15, 165
PAN_START = 90.0
TILT_START = 90.0

# ----- Web server -----
HOST = "0.0.0.0"
PORT = 8080

# ----- Bitrate (kbps) -----
START_KBPS = 2500
MIN_KBPS   = 2500
MAX_KBPS   = 4000
ADD_FRAMERATE_HINT = True

# ----- Authentication -----
AUTH_ENABLED = True
AUTH_USERS = {
    "admin": "admin"
}
SESSION_COOKIE_NAME = "ps_session"

# ----- Simulation Settings -----
SIM_PC_IP = "192.168.1.101"
SIM_CMD_PORT = 9999
SIM_VIDEO_PORT = 9998

# ----- ZOOM SETTINGS -----
ENABLE_ZOOM = True
ZOOM_START_DISTANCE_CM = 200.0  # Start zooming when the target is farther than this distance
MAX_ZOOM_LEVEL = 3.0            # Maximum zoom level (e.g., 3x)
