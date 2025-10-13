from pathlib import Path
import asyncio
import secrets
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

import config
from sdp_utils import set_bitrate_hints
from webrtc_track import FrameSource, WebRTCVideoTrack

# --------- Auth helpers ---------

def _get_sessions(app) -> dict:
    # token -> username
    return app.setdefault("sessions", {})

def _current_user(request) -> str | None:
    token = request.cookies.get(config.SESSION_COOKIE_NAME)
    if not token:
        return None
    return _get_sessions(request.app).get(token)

def _require_auth(request) -> bool:
    if not config.AUTH_ENABLED:
        return True
    return _current_user(request) is not None

# --------- App factory & routes ---------

def create_app(frame_source: FrameSource, stop_event) -> web.Application:
    app = web.Application()
    app["frame_source"] = frame_source
    app["pcs"] = set()          # all peer connections created
    app["connected"] = set()    # only those in "connected" state
    app["stop_event"] = stop_event
    app["sessions"] = {}        # in-memory session store

    # Web UI
    app.router.add_get("/", index)
    app.router.add_get("/login", login_page)
    app.router.add_post("/login", login_post)
    app.router.add_get("/logout", logout)

    # WebRTC signaling
    app.router.add_post("/offer", offer)

    # Live stats (viewer count)
    app.router.add_get("/stats", stats)

    # static files
    static_path = Path(__file__).parent / "static"
    app.router.add_static("/static/", path=static_path, name="static")

    app.on_shutdown.append(on_shutdown)
    return app

async def index(request: web.Request) -> web.StreamResponse:
    if config.AUTH_ENABLED and not _require_auth(request):
        raise web.HTTPFound("/login")
    index_path = Path(__file__).parent / "static" / "index.html"
    return web.FileResponse(index_path)

async def login_page(request: web.Request) -> web.StreamResponse:
    # If already logged in, go home
    if config.AUTH_ENABLED and _require_auth(request):
        raise web.HTTPFound("/")
    login_path = Path(__file__).parent / "static" / "login.html"
    return web.FileResponse(login_path)

async def login_post(request: web.Request) -> web.Response:
    if not config.AUTH_ENABLED:
        raise web.HTTPFound("/")

    username = password = None
    ctype = request.headers.get("Content-Type", "")
    if "application/json" in ctype:
        data = await request.json()
        username = (data.get("username") or "").strip()
        password = (data.get("password") or "").strip()
    else:
        data = await request.post()
        username = (data.get("username") or "").strip()
        password = (data.get("password") or "").strip()

    if not username or not password:
        return web.json_response({"ok": False, "error": "Missing credentials"}, status=400)

    real_pw = config.AUTH_USERS.get(username)
    if real_pw is None or real_pw != password:
        return web.json_response({"ok": False, "error": "Invalid username or password"}, status=401)

    token = secrets.token_urlsafe(32)
    _get_sessions(request.app)[token] = username

    resp = web.json_response({"ok": True, "redirect": "/"})
    resp.set_cookie(
        config.SESSION_COOKIE_NAME,
        token,
        httponly=True,
        secure=False,     # set True if serving over HTTPS
        samesite="Lax",
        max_age=60 * 60 * 12,
        path="/",
    )
    return resp

async def logout(request: web.Request) -> web.Response:
    token = request.cookies.get(config.SESSION_COOKIE_NAME)
    if token:
        _get_sessions(request.app).pop(token, None)
    resp = web.HTTPFound("/login")
    resp.set_cookie(config.SESSION_COOKIE_NAME, "", max_age=0, path="/")
    raise resp

async def _try_set_sender_bitrates(pc: RTCPeerConnection) -> None:
    # (Optional best-effort implementation you may already have.)
    # Left as-is / no-op for brevity.
    pass

async def offer(request: web.Request) -> web.Response:
    # Protect signaling
    if config.AUTH_ENABLED and not _require_auth(request):
        return web.json_response({"error": "unauthorized"}, status=401)

    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    request.app["pcs"].add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        state = pc.connectionState
        print("Connection state:", state)
        if state == "connected":
            request.app["connected"].add(pc)
        if state in ("failed", "closed", "disconnected"):
            request.app["connected"].discard(pc)
            await pc.close()
            request.app["pcs"].discard(pc)

    # Video track
    track = WebRTCVideoTrack(request.app["frame_source"], target_fps=config.TARGET_FPS)
    pc.addTrack(track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # (optional) try sender parameters
    # await _try_set_sender_bitrates(pc)

    # Patch SDP with robust hints
    patched_sdp = set_bitrate_hints(
        pc.localDescription.sdp,
        start_kbps=config.START_KBPS,
        min_kbps=config.MIN_KBPS,
        max_kbps=config.MAX_KBPS,
        framerate=config.TARGET_FPS,
        add_framerate_hint=config.ADD_FRAMERATE_HINT,
    )

    # Debug negotiated codecs
    print("\n--- NEGOTIATED SDP (video rtpmap) ---")
    for line in patched_sdp.splitlines():
        if line.startswith("a=rtpmap:") and any(k in line for k in ("VP8", "H264", "VP9", "AV1")):
            print(line)
    print("-------------------------------------\n")

    return web.json_response({"sdp": patched_sdp, "type": pc.localDescription.type})

async def stats(request: web.Request) -> web.Response:
    """
    Returns live viewer count. We count only peers in 'connected' state.
    """
    if config.AUTH_ENABLED and not _require_auth(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    viewers = len(request.app["connected"])
    resp = web.json_response({"viewers": viewers})
    # prevent caches from lying
    resp.headers["Cache-Control"] = "no-store"
    return resp

async def on_shutdown(app: web.Application) -> None:
    coros = [pc.close() for pc in app["pcs"]]
    await asyncio.gather(*coros, return_exceptions=True)
    app["pcs"].clear()
    app["connected"].clear()
    app["stop_event"].set()