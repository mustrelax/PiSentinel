# ----- Video / camera -----
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# ----- Model & tracking -----
MODEL_PATH = "./models/pisentinelv2n320_p2_ncnn_model"
FOCAL_LENGTH = 437.5
KNOWN_WIDTH_CM = 16.0
DEAD_ZONE_RADIUS = 7      # 7px is better for stability

# ----- PID -----
#PID_PAN = dict(Kp=0.06, Ki=0.01, Kd=0.005)
#PID_TILT = dict(Kp=0.07, Ki=0.01, Kd=0.005)
PID_PAN = dict(Kp=0.012, Ki=0.003, Kd=0.001)  # soft pid
PID_TILT = dict(Kp=0.012, Ki=0.003, Kd=0.001) # soft pid
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
SIM_CAPTURE_INDEX = 8

# ----- Logging Settings -----
LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
LOG_FILE = "pisentinel.log"

# ----- ZOOM SETTINGS -----
ENABLE_ZOOM = True
ZOOM_START_DISTANCE_CM = 200.0
MAX_ZOOM_LEVEL = 3.0 

# ----- OPTIMIZATION SETTINGS -----
ROI_ENABLED = True        # roi tracking
ROI_SIZE = 160            # crop size
ROI_MARGIN = 50           # margin
ADAPTIVE_INFERENCE = True # skip frames if confident
ADAPTIVE_SKIP_FRAMES = 2  # skip limit

# ----- DETECTION TUNING -----
LOCK_CONF_THRESHOLD = 0.25  # Keep strict lock in ROI mode
SEARCH_CONF_THRESHOLD = 0.30 # Reduced false positives in complex backgrounds (was 0.30)
GAMMA_CORRECTION = 1.2      # >1.0 brightens shadows, <1.0 darkens
CLAHE_CLIP_LIMIT = 4.0      # Increased contrast for difficult backgrounds
ZOOM_SEARCH_ENABLED = True  # Enable Smart Zoom (Global Mode only)

# ----- TRACKING LIMITS -----
MIN_TRACK_SIZE = 3        # 5'ti bu, 3 yaptım


# ----- HUD SETTINGS -----
HUD_COLOR_PRIMARY = (0, 255, 217)  # cyan
HUD_COLOR_WARNING = (0, 165, 255)  # orange
HUD_COLOR_DANGER  = (0, 0, 255)    # red
HUD_OPACITY = 0.8 

# ----- SENTRY MODE -----
SENTRY_MODE_ENABLED = True
IDLE_TIMEOUT = 5.0        # wait before global scan
LOCAL_SEARCH_TIMEOUT = 3.0 # local search time
LOCAL_SCAN_ANGLE = 30     # local scan angle
SENTRY_SCAN_ANGLE = 50    # global scan angle
SENTRY_SCAN_SPEED = 0.01  # scan speed

# ===== ADVANCED THREAT ANALYSIS =====



# Threat zone boundaries
THREAT_ZONE_RED_TTI = 5.0           # critical seconds
THREAT_ZONE_YELLOW_TTI = 15.0       # moderate seconds
THREAT_ZONE_RED_DISTANCE = 150      # critical distance
THREAT_ZONE_YELLOW_DISTANCE = 300   # moderate distance
THREAT_ZONE_RED_APPROACH = 8.0      # critical approach
THREAT_ZONE_YELLOW_APPROACH = 3.0   # moderate approach

# Adaptive lock bonus (replaces static LOCK_BONUS = 1000)
LOCK_BONUS_RED = 200.0       # strong lock
LOCK_BONUS_YELLOW = 100.0    # medium lock
LOCK_BONUS_GREEN = 50.0      # weask lock
LOCK_BONUS_DECAY_FRAMES = 300  # decay threshold

# Switch thresholds
SWITCH_THRESHOLD_UPGRADE = 100.0     # hard to switch
SWITCH_THRESHOLD_SAME_RED = 60.0    # hard switch within red
SWITCH_THRESHOLD_SAME_OTHER = 120.0  # çok zor switch
SWITCH_THRESHOLD_DOWNGRADE = 150.0  # downgrade hard

# Re-assessment intervals (frame count at 30fps)
REASSESS_INTERVAL_SINGLE = 90       # 3 seconds - single target
REASSESS_INTERVAL_SMALL = 60        # 2 seconds - small group
REASSESS_INTERVAL_MEDIUM = 45       # 1.5 seconds - medium swarm
REASSESS_INTERVAL_LARGE = 30        # 1 second - large swarm

# Swarm detection
SWARM_DETECTION_RADIUS = 200        # pixels - for leader detection
SWARM_LEADER_MIN_TRACKS = 3         # Minimum tracks to qualify as swarm

# Trajectory prediction
TRAJECTORY_PREDICTION_TIME = 3.0    # seconds ahead
TRAJECTORY_PREDICTION_BONUS = 50.0  # score if approaching

# Acceleration detection
ACCEL_THRESHOLD_AGGRESSIVE = 5.0    # px/frame increase for bonus
ACCEL_BONUS_FACTOR = 1.5            # multiply score
DECEL_PENALTY_FACTOR = 0.8          # multiply score

# Scenario detection thresholds
SCENARIO_SMALL_GROUP_MAX = 4        # Up to 4 = small group
SCENARIO_MEDIUM_SWARM_MAX = 10      # 5-10 = medium swarm, 10+ = large




SWARM_LEADER_DETECTION_ENABLED = True