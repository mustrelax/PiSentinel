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

## UE5 simulation (digital twin)

The system can be developed and tested without physical hardware against an Unreal Engine
5.5 digital twin. A virtual pan-tilt camera actor mirrors the real servo angles in real
time (received over UDP), and its rendered view is fed back to the edge node through an
HDMI/USB capture card.

```
  edge node (main.py --mode simulation)
        │  pan/tilt angles, raw UDP string e.g. "45.5,30.2"   ┌────────────────────────┐
        ├───────────────────────────────────────────────────▶│ UE5: BP_PanTilt_Kamera  │
        │                  (SIM_PC_IP : SIM_CMD_PORT)         │ rotates virtual camera  │
        │  rendered view  ◀──────── HDMI/USB capture card ◀───┴────────────────────────┘
        ▼
  detection + tracking + control loop
```

> The full **Unreal Engine 5.5** project (~6 GB) is hosted outside this repository.
> [**Download the UE5 project**](https://drive.google.com/file/d/1Mg_ktoVmQcpzYJ01UG4jg2hHnxZINLYP/view?usp=sharing)

### UE5-side setup
- **Unreal Engine 5.5**.
- **SocketIOClient-Unreal** plugin (https://github.com/getnamo/SocketIOClient-Unreal) —
  the `BP_PanTilt_Kamera` actor parses the UDP pan/tilt string in its `Event Tick` and
  applies it to the camera Rotator. (Angles are pre-corrected as `90 − angle` on the
  sender so they can be applied directly.)
- **Landscapes Pack** (DigitalTris) for the terrain/lighting — a free asset available on
  Fab; install it into the project from there.
- One **HDMI/USB capture card** per simulated node feeds the rendered frame to the edge
  node.

Run the edge node in simulation mode:
```bash
python main.py --mode simulation
```

### Simulation settings (`config.py`)
| Setting | Default | Meaning |
|---|---|---|
| `SIM_PC_IP` | `192.168.1.101` | IP of the PC running UE5 (where UDP packets are sent) |
| `SIM_CMD_PORT` | `9999` | UDP port the UE5 blueprint listens on for pan/tilt angles |
| `SIM_VIDEO_PORT` | `9998` | Secondary UDP port (reserved for video-side signalling) |
| `SIM_CAPTURE_INDEX` | `0` | OpenCV index of this node's HDMI/USB capture card |
| `SIM_KNOWN_WIDTH_CM` | `260.0` | Empirical constant for range estimation (absorbs UE5 mesh size + FOV + capture focal effect). **Calibrate:** put one drone 100 m north of node 1, read the reported range, then `SIM_KNOWN_WIDTH_CM *= 100 / reported`. |
| `LATENCY_COMPENSATION_ENABLED` | `True` | Enable in sim (HDMI capture adds delay); disable for the real camera |
| `LATENCY_COMPENSATION_MS` | `200` | Estimated capture-card + pipeline delay (ms) |

### Dual-node (split-screen)
One 1920×1080 split-screen render is divided in half: left → node 1, right → node 2. Both
nodes run identical code; only these differ per node:

| Setting | Node 1 | Node 2 |
|---|---|---|
| `SIM_DUAL_MODE` | `True` | `True` |
| `SIM_CAPTURE_SIDE` | `"left"` | `"right"` |
| `DEVICE_NAME` | `"Sentinel-01"` | `"Sentinel-02"` |

In UE5: place **two** pan-tilt camera actors, each with its own UDP listener on a
different port; output 1920×1080 split-screen; route each half to a separate capture card.

### ENU / geolocation (world frame) — set per node
The fusion uses a shared local East-North-Up (ENU) frame. Each node must know **where it
is** and **which way it faces**:

| Setting | Default | Meaning |
|---|---|---|
| `SYSTEM_POSITION_ENU` | `{"x":0.0,"y":0.0,"z":2.5}` | This node's position in the ENU frame (meters). Origin is the shared reference point (usually node 1). Give node 2 its real offset, e.g. `{"x":54.0,"y":0.0,"z":2.5}` for ~54 m east. |
| `CAMERA_FACING_DIRECTION` | `310` | Compass bearing the camera mount faces (0 = North, 90 = East, 180 = South, 270 = West). Must match the actor's heading in UE5. |
| `GEOLOC_DIST_UNCERTAINTY` | `0.10` | Range uncertainty σ_r as a fraction of range (10 %) |
| `GEOLOC_BEARING_UNCERTAINTY_DEG` | `1.0` | Bearing uncertainty σ_b in degrees |
| `GEOLOC_MIN_UNCERTAINTY_M` | `2.0` | Floor on uncertainty (meters) so close targets still show a circle |

Point both nodes at the same `CENTRAL_SERVER_URL`, and the server fuses their reports into
a single world-frame track.

---

## Model

`models/pisentinelv3n320_p2_ncnn_model` is the active model used by `config.py`:
320×320 input, YOLOv11n with a P2 detection head, four classes (drone, bird, airplane,
helicopter).

---

## Notes

- Dashboard users are read from `central_server/users.json`, which is not versioned. Copy
  `central_server/users.example.json` to `central_server/users.json` and set your own
  passwords; the example and `config.py` contain demo credentials (`admin` / `admin`) that
  must be changed before any non-local use.
- `logs/` (CSV telemetry) and `vlogs/` (MP4 event recordings) are created at runtime.
- This is a research prototype; see the project paper for performance figures and the
  known limitations (multi-target re-identification, monocular range uncertainty, and the
  simulation-to-real gap).

---

## Hardware (bill of materials)

| Item | Qty per node | Notes |
|---|---|---|
| Raspberry Pi 5 | 1 | edge computer (CPU-only inference, no accelerator) |
| Raspberry Pi Camera Module | 1 | `real_camera` mode |
| USB HDMI capture device | 1 | `simulation` mode (UE5 output → Pi) |
| PCA9685 16-channel PWM driver (I2C) | 1 | servo driver |
| SG90 micro servo | 2 | pan (0–180°) and tilt (15–165°) |
| Pan-tilt bracket, servo power supply (5 V power bank), cabling | 1 | |
| Wi-Fi router | 1 per site | local network between nodes and server |

---

## Reproducing the results of the paper

The `paper/` folder contains the recorded session logs and the scripts that regenerate
every figure and table of the revised manuscript (IJIRA, revision R1).

| Folder / file | Content |
|---|---|
| `paper/logs/pisentinel.log` | node 1 log of the 23 May 2026 dual-node UE5 session (`[SERVO]`, `[LATENCY]`, `[STATUS]`) |
| `paper/logs/fusion.log` | server log of the same session (`[GTM] ingest`, `[FUSION_LAT]`) |
| `paper/analysis/log_analysis.py` | control, fusion-uncertainty and timing statistics and figures |
| `paper/analysis/loop_rate.py` | realised loop rate from the logging interval |
| `paper/analysis/fusion_mc.py` | Monte Carlo comparison of geolocation estimators with known truth |
| `paper/analysis/tracker_ablation/` | tracker ablation and SORT / ByteTrack / OC-SORT / BoT-SORT comparison on frozen synthetic detections |
| `paper/results/` | the JSON/CSV/PNG outputs used in the manuscript |
| `training/` | P2 model definition, training arguments, validation curves and confusion matrices |
| `weights/` | trained four-class detector (PyTorch); the ncnn export is in `models/` |
| `configs/config_sentinel02.py` | configuration of the second node in the dual-node session |

```bash
pip install numpy scipy pandas matplotlib opencv-python filterpy motmetrics boxmot==26.0.0
cd paper/analysis
python log_analysis.py ../logs ../results
python loop_rate.py ../logs/pisentinel.log ../results/loop_rate.json
python fusion_mc.py ../results
cd tracker_ablation
python make_variant.py deployed          # parameterised copy of the tracker as deployed in the sessions
python run_ablation.py . deployed ../../results 10
python summarize.py ../../results
python verify_release.py . ../../..      # released tracking/ == "unit-corrected" variant
```
`deployed/` holds the tracker files as they were when the logs were recorded;
`run_ablation.py` first checks that the parameterised copy reproduces them exactly.
Run it from a scratch directory if you do not want BoxMOT to place model files in the
current directory.

### Training

The detector was trained with Ultralytics (8.3.233 for the single-class models, 8.4.20 for
the four-class model) on Google Colab (A100):

```bash
yolo train model=training/yolo11n-p2.yaml data=<dataset>/data.yaml epochs=200 imgsz=320 batch=32 pretrained=yolo11n.pt
yolo export model=weights/pisentinel_v3_yolo11n-p2_320.pt format=ncnn imgsz=320   # FP32
```
The four-class run used `batch=256`; see `training/args.yaml`. The dataset is available from
Roboflow Universe (`relaxw0w/drone-detection-new-peksv-ayinf`).

### Changes in this release (revision R1)

- Tracker motion thresholds that were tuned in px/frame (fast/aggressive speed, stationary
  damping, divergence deletion, ghost cleanup, direction-bonus speed) are now compared with
  velocities in px/s (×`TARGET_FPS`). The recorded sessions in `paper/logs` were produced
  with the earlier version.
- The priority score converts Kalman velocities to px/frame before computing the image-plane
  time to centre, approach rate, speed and acceleration terms.
- The lost-target speed threshold of the pan-tilt controller uses the same conversion.
- The server triangulates two bearings only if the ray intersection angle is at least 15°
  and otherwise uses inverse-variance weighting.

---

## License

PiSentinel is released under the GNU Affero General Public License v3.0 (see
`LICENSE`). The detector is trained and exported with Ultralytics YOLO, which is itself
licensed under AGPL-3.0.
