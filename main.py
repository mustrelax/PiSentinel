import threading
import argparse
import logging
from aiohttp import web

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
from web_server import create_app

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

    # 3. Create hardware/IO controllers based on mode
    servo_controller = ServoController()
    sim_commander = None
    
    if args.mode == "real_camera":
        input_device = RealCameraInput()
    elif args.mode == "simulation":
        input_device = SimulationInput()
        sim_commander = SimulationCommander()
    
    input_device.start()

    # 4. Create the main tracking thread
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

    # 5. Start the web server
    app = create_app(frame_source, stop_event)
    # Suppress aiohttp access logs
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

    print(f"HTTP/WebRTC server starting at http://{config.HOST}:{config.PORT}")
    web.run_app(app, host=config.HOST, port=config.PORT, access_log=None)

if __name__ == "__main__":
    main()