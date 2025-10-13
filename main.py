import threading
import argparse
from aiohttp import web

import config
from webrtc_track import FrameSource
from tracking import tracking_loop
from server import create_app

def main():
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Pan-Tilt Tracking System")
    parser.add_argument(
        "--mode",
        type=str,
        default="real_camera",
        choices=["real_camera", "simulation"],
        help="Operating mode: 'real_camera' for physical camera, 'simulation' for Unreal Engine."
    )
    args = parser.parse_args()
    print(f"Starting in {args.mode} mode.")
    # --- ---

    frame_source = FrameSource()
    stop_event = threading.Event()

    # Pass the mode argument to the tracking thread
    t = threading.Thread(target=tracking_loop, args=(frame_source, stop_event, args.mode), daemon=True)
    t.start()

    # Start aiohttp WebRTC server
    app = create_app(frame_source, stop_event)
    print(f"HTTP/WebRTC server http://{config.HOST}:{config.PORT} starting...")
    web.run_app(app, host=config.HOST, port=config.PORT)

if __name__ == "__main__":
    main()