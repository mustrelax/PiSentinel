# ----- Video / camera -----
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# ----- Model & tracking -----
MODEL_PATH = "./models/pisentinelv3n320_p2_ncnn_model"
#FOCAL_LENGTH = 437.5
FOCAL_LENGTH = 960
#KNOWN_WIDTH_CM = 16.0
KNOWN_WIDTH_CM = 50.0
DEAD_ZONE_RADIUS = 7      # 7px is better for stability

# ----- PID + FEED-FORWARD -----
# Base PID gains — FF handles bulk tracking so PID only closes residual error
PID_PAN = dict(Kp=0.015, Ki=0.004, Kd=0.002)
PID_TILT = dict(Kp=0.015, Ki=0.004, Kd=0.002)

# Feed-forward: fraction of ideal velocity compensation (0.0=off, 1.0=perfect)
# Uses smoothed Kalman velocity to filter out servo-induced jitter
FF_GAIN_PAN = 0.85
FF_GAIN_TILT = 0.85
FF_SMOOTH_ALPHA = 0.12   # Low-pass filter for FF velocity (filters servo artifacts)
FF_MIN_SPEED = 3.0       # px/sec — don't apply FF below this (avoids jitter)

# Gain Scheduling — Kp scales with distance from center (Eq. 29-30)
GAIN_SCHEDULE_ENABLED = True
KP_CLOSE = 0.015       # r < NEAR: gentle, prevents jitter when centered
KP_FAR = 0.030         # r > FAR: aggressive catch-up for escaping drones
GAIN_SCHEDULE_NEAR = 50     # px threshold for "close"
GAIN_SCHEDULE_FAR = 150     # px threshold for "far"

# Rate limits — SG90 max ~60°/sec, so 2.5°/frame@30fps = 75°/s (safe headroom)
SERVO_RATE_LIMIT_CLOSE = 1.0   # °/frame when drone is near center
SERVO_RATE_LIMIT_FAR = 2.5     # °/frame when drone is far from center
# ----- Servo limits (degrees) -----
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 15, 165
PAN_START = 90.0
TILT_START = 90.0

# ----- Central Server -----
CENTRAL_SERVER_URL = "http://192.168.1.106:3000" 
# CENTRAL_SERVER_URL = "http://192.168.109.182:3000"
DEVICE_NAME = "Sentinel-02"

# Camera facing direction in degrees (compass bearing)
# 0 = North, 90 = East, 180 = South, 270 = West
# Example: If camera faces West, set to 270
CAMERA_FACING_DIRECTION = 50  # East

# ----- Geolocation (ENU, meters) -----
# Local East/North/Up frame. x=East, y=North, z=Up. Origin is the reference
# point shared by all units in this deployment (usually the first camera).
SYSTEM_POSITION_ENU = {"x": -235.1, "y": 55.4, "z": -11.1}
GEOLOC_DIST_UNCERTAINTY = 0.10          # sigma_r as fraction of range (monocular)
GEOLOC_BEARING_UNCERTAINTY_DEG = 1.0    # sigma_b in degrees (servo + optical)
GEOLOC_MIN_UNCERTAINTY_M = 2.0          # floor, so tiny ranges still have a visible circle

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
SIM_CMD_PORT = 9997
SIM_VIDEO_PORT = 9996
SIM_CAPTURE_INDEX = 0

# Dual-node simulation: tek bir 1920x1080 capture'ı yarıya bölüp
# sol yarıyı PiSentinel1, sağ yarıyı PiSentinel2 olarak kullanma modu.
# Her iki node'a aynı kod yüklenir, sadece SIM_CAPTURE_SIDE farklı olur.
SIM_DUAL_MODE = True            # True => split-screen capture'dan bir yarıyı al
SIM_CAPTURE_SIDE = "right"        # "left" veya "right" — PiSentinel1=left, PiSentinel2=right

# ----- Logging Settings -----
LOG_LEVEL = "INFO"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
LOG_FILE = "pisentinel.log"

# ----- ZOOM SETTINGS -----
ENABLE_ZOOM = True
ZOOM_START_DISTANCE_CM = 200.0
MAX_ZOOM_LEVEL = 3.0 

# ----- OPTIMIZATION SETTINGS -----
ROI_ENABLED = False        # Enabled for single drone tracking
ROI_SIZE = 160            # crop size
ROI_MARGIN = 50           # margin
ADAPTIVE_INFERENCE = True # skip frames if confident
ADAPTIVE_SKIP_FRAMES = 2  # skip limit



# ----- DETECTION TUNING -----
LOCK_CONF_THRESHOLD = 0.25  # Keep strict lock in ROI mode
SEARCH_CONF_THRESHOLD = 0.25 # Lowered for detection against complex backgrounds (was 0.30)
GAMMA_CORRECTION = 1.2      # >1.0 brightens shadows, <1.0 darkens
CLAHE_CLIP_LIMIT = 4.0      # Increased contrast for difficult backgrounds
ZOOM_SEARCH_ENABLED = True  # Enable Smart Zoom (Global Mode only)

# ----- SIMULATION PROFILE (UE5 quick testing) -----
# Keeps simulation tuning isolated so real-camera parameters stay untouched.
LATENCY_COMPENSATION_ENABLED = True   # Enable for sim (HDMI delay), disable for real camera
LATENCY_COMPENSATION_MS = 200         # Estimated capture card + pipeline delay in ms
SIM_ENABLE_PREPROCESSING = False
SIM_LIGHTWEIGHT_PREPROCESSING = False
SIM_CLAHE_CLIP_LIMIT = 2.0
SIM_SHARPEN_STRENGTH = 0.15
SIM_GAMMA_CORRECTION = 1.05
SIM_LOCK_CONF_THRESHOLD = 0.25
SIM_SEARCH_CONF_THRESHOLD = 0.23
SIM_ADAPTIVE_SKIP_FRAMES = 2
SIM_KNOWN_WIDTH_CM = 260.0

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

# Adaptive lock bonus — tuned for 0-1000 score range
LOCK_BONUS_RED = 120.0       # strong lock on RED zone target
LOCK_BONUS_YELLOW = 70.0     # medium lock on YELLOW zone target
LOCK_BONUS_GREEN = 35.0      # light lock on GREEN zone target
LOCK_BONUS_DECAY_FRAMES = 300  # decay threshold

# Switch thresholds — tuned for 0-1000 score range
SWITCH_THRESHOLD_UPGRADE = 40.0      # easy upgrade to more dangerous zone
SWITCH_THRESHOLD_SAME_RED = 60.0     # moderate switch within red
SWITCH_THRESHOLD_SAME_OTHER = 75.0   # stay on non-red target
SWITCH_THRESHOLD_DOWNGRADE = 100.0   # hard downgrade

# Re-assessment intervals (frame count at 30fps)
REASSESS_INTERVAL_SINGLE = 150      # 5 seconds - single target
REASSESS_INTERVAL_SMALL = 150       # 5 seconds - small group
REASSESS_INTERVAL_MEDIUM = 120      # 4 seconds - medium swarm
REASSESS_INTERVAL_LARGE = 90        # 3 seconds - large swarm

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

# Threat score EMA — asymmetric: fast rise, slow decay ("sticky high")
THREAT_SCORE_RISE_ALPHA = 0.15   # how fast score climbs when threat increases
THREAT_SCORE_DECAY_ALPHA = 0.04  # how slowly score falls when threat reduces
SWITCH_MARGIN_MULTIPLIER = 1.0  # was 1.25 — was silently inflating all thresholds by 25%
SWITCH_CONFIRMATIONS_REQUIRED = 2

# Normalization maximums for velocity/approach components
APPROACH_RATE_MAX = 20.0   # px/frame — saturates approach_norm at 1.0
VELOCITY_NORM_MAX = 20.0   # px/frame — saturates velocity_norm at 1.0

# Scenario-aware minimum lock duration before reassessment
MIN_LOCK_DURATION_SINGLE = 60
MIN_LOCK_DURATION_SMALL = 90
MIN_LOCK_DURATION_MEDIUM = 120
MIN_LOCK_DURATION_LARGE = 120

# Swarm-group density detection radius
SWARM_DENSITY_RADIUS = 220
# SWARM_DENSITY_WEIGHT removed — density is now a normalized component in scenario weights

# Performance: update 3D estimation for passive tracks less frequently
PASSIVE_3D_UPDATE_INTERVAL = 3
