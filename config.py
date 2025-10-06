# ----- Video / camera -----
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 25

# ----- Model & tracking -----
MODEL_PATH = "model_ncnn_model320"
TRACKER_CONFIG = "bytetrack.yaml"
FOCAL_LENGTH = 437.5
KNOWN_WIDTH_CM = 16.0
DEAD_ZONE_RADIUS = 20

# ----- PID -----
PID_PAN = dict(Kp=0.06, Ki=0.01, Kd=0.005)
PID_TILT = dict(Kp=0.07, Ki=0.01, Kd=0.005)

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
