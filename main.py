import threading
from aiohttp import web

import config
from webrtc_track import FrameSource
from tracking import tracking_loop
from server import create_app

def main():
    frame_source = FrameSource()
    stop_event = threading.Event()

    # Start tracking thread
    t = threading.Thread(target=tracking_loop, args=(frame_source, stop_event), daemon=True)
    t.start()

    # Start aiohttp WebRTC server
    app = create_app(frame_source, stop_event)
    print(f"HTTP/WebRTC server http://{config.HOST}:{config.PORT} starting...")
    web.run_app(app, host=config.HOST, port=config.PORT)

if __name__ == "__main__":
    main()
