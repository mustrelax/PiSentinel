# PiSentinel

Real-time, vision-based drone detection and pan-tilt tracking system that runs on a
Raspberry Pi 5 edge node and reports to a central server. PiSentinel detects UAVs with a
YOLOv11n model (P2 detection head, exported to ncnn), tracks them with an adaptive
Kalman filter + Hungarian-matching tracker, physically follows the locked target with a
PID + feed-forward pan-tilt controller, and streams low-latency video to a web command
and control dashboard over WebRTC (Mediasoup SFU). Multiple nodes can report
East-North-Up (ENU) world-frame estimates that the server fuses by bearing-only
triangulation and inverse-variance weighting.

> Research prototype developed as a final-year project at Marmara University. Reported
> performance is based on Unreal Engine 5 simulation validation; large-scale outdoor
> field validation remains future work.

---

## System architecture

PiSentinel is split into two kinds of node:

- **Edge unit (Raspberry Pi 5)** — `main.py` and the Python modules. Handles camera
  capture, detection, tracking, threat analysis, ENU geolocation and pan-tilt motor
  control. Runs independently of network latency.
- **Central server (laptop/PC)** — `central_server/` (Node.js). Mediasoup SFU video
  distribution, Socket.IO telemetry, multi-node track fusion, logging, and the operator
  dashboard.

```
┌────────────────────┐        Socket.IO (telemetry)        ┌─────────────────────┐
│  Edge unit (Pi 5)  │  ─────────────────────────────────▶ │   Central server    │
│  main.py           │        WebRTC (video, Mediasoup)    │   central_server/    │
│  detection+tracking│  ◀───────────────────────────────── │   fusion + dashboard │
│  pan-tilt control  │        peer ENU geolocation         │   http://host:3000  │
└────────────────────┘                                     └─────────────────────┘
```

---

## Repository layout

```
.
├── main.py                 # Edge-node entry point (real_camera | simulation)
├── config.py               # All edge-node settings (model, PID, server URL, geolocation…)
├── io_devices.py           # Camera / capture-card input, servo driver, UE5 sim commander
├── tracking_thread.py      # Glue thread: capture → track → control loop
├── webrtc_utils.py         # Shared frame source for the WebRTC producer
├── central_connector.py    # Socket.IO client + aiortc → Mediasoup video producer
├── tracking/               # Tracker, matcher, threat analyzer, geolocation, loggers
│   ├── controller.py        multi_tracker.py   matcher.py   track.py
│   ├── threat_analyzer.py   geolocation.py     geolocation_manager.py
│   └── track_logger.py      video_logger.py
├── models/                 # Active ncnn model (pisentinelv3n320_p2_ncnn_model)
├── central_server/         # Node.js Mediasoup SFU + dashboard (see its own README.md)
├── visualization/, tests/
└── requirements.txt        # Python dependencies for the edge node
```

---

## Prerequisites

### Edge unit (Raspberry Pi 5)
- Python 3.10+
- The Python packages in `requirements.txt`
- Pan-tilt hardware: PCA9685 I2C servo driver + 2× SG90 servos (only needed in
  `real_camera` mode; `simulation` mode runs without hardware)
- Pi Camera Module (real_camera mode) **or** an HDMI/USB capture device (simulation mode)

### Central server (laptop/PC)
- Node.js 16+ (Mediasoup compiles native modules — see `central_server/README.md` for the
  Windows build-tools note)

---

## Installation

### 1. Central server
```bash
cd central_server
npm install        # may take a few minutes (compiles Mediasoup)
```

### 2. Edge unit
```bash
pip install -r requirements.txt
```
The Adafruit hardware packages in `requirements.txt` are only required on the Raspberry Pi
(`real_camera` mode). On a development PC running `simulation` mode you can skip them.

---

## Configuration

All edge-node settings live in `config.py`. The most important ones:

| Setting | Meaning |
|---|---|
| `MODEL_PATH` | Active ncnn model (default `./models/pisentinelv3n320_p2_ncnn_model`) |
| `CENTRAL_SERVER_URL` | URL of the central server, e.g. `http://192.168.1.105:3000` |
| `DEVICE_NAME` | Unique node id, e.g. `Sentinel-01` |
| `CAMERA_FACING_DIRECTION` | Compass bearing the camera faces (0 = North, 90 = East …) |
| `SYSTEM_POSITION_ENU` | This node's position in the shared ENU world frame (meters) |
| `SIM_DUAL_MODE`, `SIM_CAPTURE_SIDE` | Dual-node split-screen simulation (see below) |
| `AUTH_USERS` | Dashboard login (default `admin` / `admin`) |

Set `CENTRAL_SERVER_URL` to the IP of the machine running the central server, and make
sure both machines are on the same network.

---

## Running

### 1. Start the central server
```bash
cd central_server
npm start
```
It listens on `http://0.0.0.0:3000`. Open `http://localhost:3000` (or the host's LAN IP
from another device) and log in with `admin` / `admin`.

### 2. Start an edge unit

Real camera + pan-tilt hardware on the Raspberry Pi:
```bash
python main.py --mode real_camera
```

Unreal Engine 5 simulation (HDMI/USB capture card, no servo hardware required):
```bash
python main.py --mode simulation
```

The edge unit connects to `CENTRAL_SERVER_URL`, begins detecting and tracking, and its
stream and telemetry appear on the dashboard.

---

## Dual-node simulation

To develop and test multi-node fusion from a single Unreal Engine instance, a single
1920×1080 split-screen capture is divided in half: the left half feeds node 1, the right
half feeds node 2. Both nodes run identical code; only two config values differ:

```python
SIM_DUAL_MODE   = True
SIM_CAPTURE_SIDE = "left"   # "left" => Sentinel-01,  "right" => Sentinel-02
```

Give each node a distinct `DEVICE_NAME`, `SYSTEM_POSITION_ENU` and
`CAMERA_FACING_DIRECTION`, point both at the same `CENTRAL_SERVER_URL`, and the server
fuses their reports into a single world-frame track.

---

## Model

`models/pisentinelv3n320_p2_ncnn_model` is the active model used by `config.py`:
320×320 input, YOLOv11n with a P2 detection head, four classes (drone, bird, airplane,
helicopter).

---

## Notes

- Default dashboard credentials are `admin` / `admin` (`central_server/users.json` and
  `config.py`). Change them before any non-local use.
- `logs/` (CSV telemetry) and `vlogs/` (MP4 event recordings) are created at runtime.
- This is a research prototype; see the project paper for performance figures and the
  known limitations (multi-target re-identification, monocular range uncertainty, and the
  simulation-to-real gap).
