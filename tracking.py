import time
import datetime
import math
import numpy as np
import cv2
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from simple_pid import PID

import board
import busio
import socket
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

import config
from webrtc_track import FrameSource

def tracking_loop(frame_source: FrameSource, stop_event, mode: str) -> None:

    # --- Load model ---
    try:
        model = YOLO(config.MODEL_PATH)
        print(f"Model '{config.MODEL_PATH}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Initialize variables ---
    frame = None
    video_socket = None
    picam2 = None
#    ndi_recv = None

    # --- Setup based on mode ---
    if mode == "real_camera":
        # --- Camera setup (Picamera2) ---
        from picamera2 import Picamera2
        picam2 = Picamera2()
        cam_cfg = picam2.create_video_configuration(
            main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT), "format": "RGB888"},
            raw={"size": picam2.sensor_resolution}
        )
        picam2.configure(cam_cfg)
        picam2.start()
        print("Picamera2 started.")
        time.sleep(1)
    
    elif mode == "simulation":
        CAPTURE_INDEX = 9
        print(f"Attempting to open capture card at index {CAPTURE_INDEX}...")

        video_capture = cv2.VideoCapture(CAPTURE_INDEX, cv2.CAP_V4L2)
        
        if not video_capture.isOpened():
            print(f"FATAL ERROR: Capture card could not be opened at index {CAPTURE_INDEX}.")
            return
        
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        video_capture.set(cv2.CAP_PROP_FPS, 60)
        
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print(f"Capture card successfully initialized. Actual settings: {int(width)}x{int(height)} @ {fps:.2f} FPS")

    # --- Add command sender for simulation mode ---
    command_socket = None
    if mode == "simulation":
        command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"Will send commands to Unreal at {config.SIM_PC_IP}:{config.SIM_CMD_PORT}")

    # --- PCA9685 servo setup ---
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=0x40)
    pca.frequency = 50

    pan_servo = servo.Servo(pca.channels[0])
    tilt_servo = servo.Servo(pca.channels[1])

    pan_angle = float(config.PAN_START)
    tilt_angle = float(config.TILT_START)
    pan_servo.angle = pan_angle
    tilt_servo.angle = tilt_angle

    # --- PID controllers ---
    pid_pan = PID(**config.PID_PAN, setpoint=config.FRAME_WIDTH // 2)
    pid_pan.output_limits = (-3, 3)   # deg/frame

    pid_tilt = PID(**config.PID_TILT, setpoint=config.FRAME_HEIGHT // 2)
    pid_tilt.output_limits = (-3, 3)

    # --- Kalman filter (x, y, vx, vy) ---
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.zeros(4)
    kf.F = np.array([[1,0,1,0],
                     [0,1,0,1],
                     [0,0,1,0],
                     [0,0,0,1]], dtype=float)
    kf.H = np.array([[1,0,0,0],
                     [0,1,0,0]], dtype=float)
    kf.P *= 1000.0
    kf.R = np.array([[5,0],
                     [0,5]], dtype=float)
    kf.Q = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,10,0],
                     [0,0,0,10]], dtype=float)

    tracked_id = None
    lost_counter = 0
    fps_buffer = []
    FPS_BUFFER_SIZE = 30
    
    try:
        while not stop_event.is_set():
            t0 = time.perf_counter()

            # --- Get frame based on mode ---
            if mode == "real_camera":
                frame = picam2.capture_array()
            elif mode == "simulation":
                ret, frame = video_capture.read()
                if not ret:
                    print("Warning: Could not read frame from capture card.")
                    time.sleep(0.01)
                    continue            
            
            results = model.track(frame, persist=True, verbose=False, tracker=config.TRACKER_CONFIG, conf=0.25)
            annotated = frame.copy()

            boxes = None
            track_ids = None
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                for i in range(len(boxes)):
                    if track_ids[i] != tracked_id:
                        x1, y1, x2, y2 = boxes[i]
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if tracked_id is None:
                # Lock to the target closest to horizontal center
                if boxes is not None:
                    cx_screen = config.FRAME_WIDTH // 2
                    best_id, best_dist = None, float("inf")
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i]
                        cx = (x1 + x2) // 2
                        d = abs(cx - cx_screen)
                        if d < best_dist:
                            best_dist, best_id = d, track_ids[i]
                    if best_id is not None:
                        tracked_id = best_id
                        lost_counter = 0
                        for i in range(len(boxes)):
                            if track_ids[i] == tracked_id:
                                x1, y1, x2, y2 = boxes[i]
                                kf.x = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, 0.0, 0.0])
                                pid_pan.reset()
                                pid_tilt.reset()
                                break
                        print(f"Locked on new target ID: {tracked_id}")

            else:
                kf.predict()
                target_found = False

                if boxes is not None:
                    for i in range(len(boxes)):
                        if track_ids[i] == tracked_id:
                            target_found = True
                            x1, y1, x2, y2 = boxes[i]

                            # --- ADAPTIVE KALMAN FILTER ---
                            
                            # Get the confidence score for this specific detection.
                            confidence_score = results[0].boxes.conf.cpu().numpy()[i]

                            # (2) Create a temporary, adaptive measurement noise matrix (R).
                            adaptive_R = kf.R.copy()

                            # (3) Apply the formula from the paper: R_tilde = (1 - c) * R
                            adaptive_R *= (1.0 - confidence_score)
                            
                            # (4) Update the Kalman Filter using the new, adaptive noise matrix.
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            kf.update(z=np.array([cx, cy]), R=adaptive_R)
                            
                            # --- ADAPTIVE KALMAN FILTER ---
                            
                            pixel_w = max(1, x2 - x1)
                            distance_cm = (config.KNOWN_WIDTH_CM * config.FOCAL_LENGTH) / pixel_w

                            sx, sy = int(kf.x[0]), int(kf.x[1])
                            dx = sx - (config.FRAME_WIDTH // 2)
                            dy = sy - (config.FRAME_HEIGHT // 2)

                            # Pinhole projection to centimeters
                            X_cm = (dx * distance_cm) / config.FOCAL_LENGTH
                            Y_cm = -(dy * distance_cm) / config.FOCAL_LENGTH
                            Z_cm = distance_cm

                            # PID + servos
                            radius = (dx * dx + dy * dy) ** 0.5
                            if radius < config.DEAD_ZONE_RADIUS:
                                pan_corr = 0.0
                                tilt_corr = 0.0
                            else:
                                pan_corr = pid_pan(sx)
                                tilt_corr = pid_tilt(sy)

                            pan_angle += pan_corr
                            tilt_angle -= tilt_corr

                            pan_angle = max(config.PAN_MIN, min(config.PAN_MAX, pan_angle))
                            tilt_angle = max(config.TILT_MIN, min(config.TILT_MAX, tilt_angle))

                            pan_servo.angle = pan_angle
                            tilt_servo.angle = tilt_angle

                            # If in simulation mode, also send commands to Unreal
                            if mode == "simulation":
                                sim_pan = 90.0 - pan_angle
                                sim_tilt = 90.0 - tilt_angle

                                message = f"{sim_pan},{sim_tilt}"
                                command_socket.sendto(message.encode(), (config.SIM_PC_IP, config.SIM_CMD_PORT))

                            # World transform for display
                            pan_rad = math.radians(pan_angle - 90.0)
                            tilt_rad = math.radians(tilt_angle - 90.0)
                            x_cam, y_cam, z_cam = X_cm, Y_cm, Z_cm
                            x_panned = z_cam * math.sin(pan_rad) + x_cam * math.cos(pan_rad)
                            z_panned = z_cam * math.cos(pan_rad) - x_cam * math.sin(pan_rad)
                            Xw = x_panned
                            Yw = y_cam * math.cos(tilt_rad) - z_panned * math.sin(tilt_rad)
                            Zw = y_cam * math.sin(tilt_rad) + z_panned * math.cos(tilt_rad)
                            dist_th = math.sqrt(Xw**2 + Yw**2 + Zw**2)

                            # Draw overlay
                            score_text = f"Score: {confidence_score:.2f}"
                            text_position = (x1, y2 + 20) 
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(annotated, score_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255, 255), 2)
                            cv2.circle(annotated, (sx, sy), 5, (0, 0, 255), -1)
                            cv2.putText(annotated, f"World XYZ: {Xw:.1f}, {Yw:.1f}, {Zw:.1f} cm",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(annotated, f"Direct Dist: {z_cam:.1f} cm | Theoretical: {dist_th:.1f} cm",
                                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            break

                if target_found:
                    lost_counter = 0
                else:
                    lost_counter += 1
                    if lost_counter > 15:
                        print(f"Target ID {tracked_id} lost. Searching...")
                        tracked_id = None

            # FPS overlay
            t1 = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, (t1 - t0))
            fps_buffer.append(inst_fps)
            if len(fps_buffer) > FPS_BUFFER_SIZE:
                fps_buffer.pop(0)
            avg_fps = sum(fps_buffer) / len(fps_buffer)
            cv2.putText(annotated, f"Avg FPS: {avg_fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Time
            timestamp_str = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            cv2.putText(annotated, timestamp_str, (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Publish frame
            frame_source.set(annotated)

    except Exception as e:
        print("Tracking loop error:", e)
    finally:
        try:
            print("\nCentering servos and releasing channels...")
            try:
                pan_servo.angle = config.PAN_START
                tilt_servo.angle = config.TILT_START
                time.sleep(0.5)
            except Exception:
                pass
            pca.deinit()

            try:
                if video_capture is not None:
                    print("Releasing capture card...")
                    video_capture.release()
            except Exception as e:
                print(f"Error releasing capture card: {e}")
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Tracking loop terminated.")