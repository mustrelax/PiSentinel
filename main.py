import threading
import argparse
import logging
# from aiohttp import web

import config
from webrtc_utils import FrameSource
from tracking import TrackingController
from io_devices import (
    RealCameraInput, 
    SimulationInput, 
    ServoController, 
    SimulationCommander
)
from tracking_thread import TrackingThread
# from web_server import create_app # Deprecated

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")


def apply_runtime_profile(mode: str, tracker):
    """Apply mode-specific runtime tuning without changing base config defaults."""
    if mode != "simulation":
        return

    tracker.ENABLE_PREPROCESSING = config.SIM_ENABLE_PREPROCESSING
    tracker.PREPROCESS_LIGHTWEIGHT = config.SIM_LIGHTWEIGHT_PREPROCESSING
    tracker.GAMMA = config.SIM_GAMMA_CORRECTION
    tracker.gamma_lut = tracker._build_gamma_lut(tracker.GAMMA)
    tracker.set_preprocessing_params(
        clip_limit=config.SIM_CLAHE_CLIP_LIMIT,
        sharpen_strength=config.SIM_SHARPEN_STRENGTH,
    )

    # Override active thresholds for this process only.
    config.LOCK_CONF_THRESHOLD = config.SIM_LOCK_CONF_THRESHOLD
    config.SEARCH_CONF_THRESHOLD = config.SIM_SEARCH_CONF_THRESHOLD
    config.ADAPTIVE_SKIP_FRAMES = config.SIM_ADAPTIVE_SKIP_FRAMES
    config.KNOWN_WIDTH_CM = config.SIM_KNOWN_WIDTH_CM

    logger.info(
        "[PROFILE] Simulation profile applied: preprocess=%s gamma=%.2f clahe=%.2f sharp=%.2f skip=%d",
        tracker.ENABLE_PREPROCESSING,
        tracker.GAMMA,
        config.SIM_CLAHE_CLIP_LIMIT,
        config.SIM_SHARPEN_STRENGTH,
        config.ADAPTIVE_SKIP_FRAMES,
    )

def main():
    parser = argparse.ArgumentParser(description="Pi-Sentinel Tracking System")
    parser.add_argument(
        "--mode",
        type=str,
        default="real_camera",
        choices=["real_camera", "simulation"],
        help="Operating mode: 'real_camera' or 'simulation'."
    )
    args = parser.parse_args()
    logger.info(f"Starting in {args.mode} mode.")

    # 1. Create shared resources
    frame_source = FrameSource()
    stop_event = threading.Event()

    # 2. Create core tracker logic
    tracker = TrackingController()
    apply_runtime_profile(args.mode, tracker)

    # 3. Create hardware/IO controllers based on mode
    servo_controller = ServoController()
    sim_commander = None
    
    if args.mode == "real_camera":
        input_device = RealCameraInput()
    elif args.mode == "simulation":
        input_device = SimulationInput()
        sim_commander = SimulationCommander()
    
    input_device.start()

    # 4. Create Central Connector and set callbacks BEFORE starting tracking thread
    # This prevents a race where tracks confirmed during startup have no callback path
    from central_connector import CentralConnector
    from tracking.geolocation_manager import GeolocationManager
    import asyncio

    connector = CentralConnector(frame_source)

    # Geolocation unit: owns world-frame state for own + peer tracks.
    # Controller feeds raw telemetry in; manager emits world estimates out to
    # the connector; connector pushes peer updates from the server back in.
    geoloc = GeolocationManager(
        system_pos=config.SYSTEM_POSITION_ENU,
        camera_facing=config.CAMERA_FACING_DIRECTION,
        device_name=config.DEVICE_NAME,
        k_range=config.GEOLOC_DIST_UNCERTAINTY,
        k_bearing_deg=config.GEOLOC_BEARING_UNCERTAINTY_DEG,
        min_uncertainty_m=config.GEOLOC_MIN_UNCERTAINTY_M,
    )
    geoloc.on_own_update = connector.send_geolocation
    connector.on_peer_geolocation = geoloc.handle_peer_update

    tracker.on_track_confirmed = lambda track_id: connector.send_alert(track_id)
    tracker.on_tracking_status = lambda track_id, status: connector.send_tracking_status(track_id, status)
    tracker.on_telemetry_update = lambda track_id, az, dist, spd: connector.send_radar_telemetry(track_id, az, dist, spd)
    tracker.on_active_telemetry = geoloc.update_own
    tracker.on_track_deleted = geoloc.drop
    tracker.on_track_log_complete = lambda summary: connector.send_track_log(summary)
    tracker.on_unit_telemetry = connector.send_unit_telemetry
    tracker.on_unit_health = connector.send_unit_health

    # 5. Create and start the main tracking thread
    # This thread glues all components together
    tracking_thread = TrackingThread(
        frame_source=frame_source,
        input_device=input_device,
        servo_controller=servo_controller,
        tracker=tracker,
        stop_event=stop_event,
        sim_commander=sim_commander  # This will be None for real_camera
    )
    tracking_thread.start()

    # 6. Start the Central Connector event loop (blocking)
    print(f"Connecting to Central Server at {config.CENTRAL_SERVER_URL}...")
    try:
        asyncio.run(connector.start())
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Finalize all active track logs before shutdown
        tracker.finalize_on_shutdown()
        stop_event.set()
        tracking_thread.join()

if __name__ == "__main__":
    main()