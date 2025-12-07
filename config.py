# ----- Video / camera -----
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# ----- Model & tracking -----
#MODEL_PATH = "model_ncnn_model320"
MODEL_PATH = "./models/pisentinelv2n320_p2_ncnn_model" 
TRACKER_CONFIG = "pytetrack.yaml"
FOCAL_LENGTH = 437.5
KNOWN_WIDTH_CM = 16.0
DEAD_ZONE_RADIUS = 7      # Increased to 7px (More stable lock)

# ----- PID -----
#PID_PAN = dict(Kp=0.06, Ki=0.01, Kd=0.005)
#PID_TILT = dict(Kp=0.07, Ki=0.01, Kd=0.005)
PID_PAN = dict(Kp=0.012, Ki=0.003, Kd=0.001)  # Softer PID
PID_TILT = dict(Kp=0.012, Ki=0.003, Kd=0.001) # Softer PID
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
ZOOM_START_DISTANCE_CM = 200.0
MAX_ZOOM_LEVEL = 3.0 

# ----- OPTIMIZATION SETTINGS -----
ROI_ENABLED = True        # Enable ROI tracking
ROI_SIZE = 160            # Size of ROI crop
ROI_MARGIN = 50           # Margin to keep track inside ROI
ADAPTIVE_INFERENCE = True # Skip frames if tracking is confident
ADAPTIVE_SKIP_FRAMES = 2  # How many frames to skip

# ----- DETECTION TUNING -----
LOCK_CONF_THRESHOLD = 0.25  # Keep strict lock in ROI mode
SEARCH_CONF_THRESHOLD = 0.40 # Reduced false positives in complex backgrounds (was 0.30)
GAMMA_CORRECTION = 1.2      # >1.0 brightens shadows, <1.0 darkens
CLAHE_CLIP_LIMIT = 4.0      # Increased contrast for difficult backgrounds
ZOOM_SEARCH_ENABLED = True  # Enable Smart Zoom (Global Mode only)

# ----- TRACKING LIMITS -----
MIN_TRACK_SIZE = 5        # Reduced noise sensitivity (was 3px)

# ----- HUD SETTINGS -----
HUD_COLOR_PRIMARY = (0, 255, 217)  # Cyan/Teal for futuristic look
HUD_COLOR_WARNING = (0, 165, 255)  # Orange
HUD_COLOR_DANGER  = (0, 0, 255)    # Red
HUD_OPACITY = 0.8 

# ----- SENTRY MODE -----
SENTRY_MODE_ENABLED = True
IDLE_TIMEOUT = 5.0        # Seconds to wait before starting GLOBAL sentry scan
LOCAL_SEARCH_TIMEOUT = 3.0 # Seconds to search locally before going global
LOCAL_SCAN_ANGLE = 30     # Degrees to scan locally
SENTRY_SCAN_ANGLE = 50    # Degrees to scan left/right from center (Global) - Wider
SENTRY_SCAN_SPEED = 0.01  # Speed of scan (radians per frame) - More aggressive

# ===== ADVANCED THREAT ANALYSIS (MILITARY-GRADE) =====

# Time-to-impact calculation
WEIGHT_TTI = 300.0

# Threat zone boundaries
THREAT_ZONE_RED_TTI = 5.0           # seconds - critical
THREAT_ZONE_YELLOW_TTI = 15.0       # seconds - moderate
THREAT_ZONE_RED_DISTANCE = 150      # pixels - critical
THREAT_ZONE_YELLOW_DISTANCE = 300   # pixels - moderate
THREAT_ZONE_RED_APPROACH = 8.0      # px/frame - critical
THREAT_ZONE_YELLOW_APPROACH = 3.0   # px/frame - moderate

# Adaptive lock bonus (replaces static LOCK_BONUS = 1000)
LOCK_BONUS_RED = 200.0       # Critical threat - strong lock
LOCK_BONUS_YELLOW = 100.0    # Moderate threat - medium lock
LOCK_BONUS_GREEN = 50.0      # Low threat - weak lock
LOCK_BONUS_DECAY_FRAMES = 300  # 30% reduction after 10 seconds

# Switch thresholds
SWITCH_THRESHOLD_UPGRADE = 50.0     # To higher threat zone
SWITCH_THRESHOLD_SAME_RED = 30.0    # Within red zone
SWITCH_THRESHOLD_SAME_OTHER = 80.0  # Within yellow/green zone
SWITCH_THRESHOLD_DOWNGRADE = 150.0  # To lower threat zone

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

# Scoring weights (base values, scenario-specific in code)
WEIGHT_DISTANCE = 100.0
WEIGHT_VELOCITY = 15.0
WEIGHT_APPROACH_RATE = 80.0
WEIGHT_SWARM_LEADER = 150.0
WEIGHT_CONFIDENCE = 5.0

# Feature flags (for rollback)
ADVANCED_THREAT_ANALYSIS_ENABLED = True
DYNAMIC_REASSESSMENT_ENABLED = True
SWARM_LEADER_DETECTION_ENABLED = True