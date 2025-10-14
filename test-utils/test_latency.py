# test_latency.py (Updated with a warm-up sequence)

import time
import socket
import cv2
import numpy as np
import config

def measure_control_latency(udp_sender_func, video_capture_func):

    print("--- CONTROL LATENCY TEST STARTED ---")
    print("Sending 100 warm-up packets...")
    for i in range(100):
        udp_sender_func(0.0, 0.0) 
        time.sleep(0.03) # 30ms delay 
    print("Warm-up complete.")
    # --- End of Warm-up Phase ---

    print("Sending test packet and starting timer...")
    

    start_time = time.perf_counter()

    udp_sender_func(0.0, 0.0) 

    timeout = start_time + 2.0 

    while time.perf_counter() < timeout:
        frame = video_capture_func()
        if frame is None:
            time.sleep(0.01)
            continue
        
        h, w, _ = frame.shape
        roi = frame[h-50:h, w-50:w] 

        if roi.mean() > 200: 
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            print(f"\n!!! FLASH DETECTED !!!")
            print(f"--> Control Loop Latency: {latency_ms:.2f} ms")
            return latency_ms

    print("\nTIMEOUT: Flash not detected.")
    return None

if __name__ == '__main__':
    print("--- SETTING UP FOR LATENCY TEST ---")
    
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    unreal_address = (config.SIM_PC_IP, config.SIM_CMD_PORT)
    
    cap = cv2.VideoCapture(8, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"FATAL: Could not open capture card at index 8. Exiting.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    print("Capture card and UDP socket are ready.")

    def send_udp_for_test(pan, tilt):
        message = f"{pan},{tilt}".encode()
        udp_socket.sendto(message, unreal_address)

    def get_frame_for_test():
        ret, frame = cap.read()
        return frame if ret else None

    print("\nWaiting 2 seconds before starting...")
    time.sleep(2)
    
    measured_latency = measure_control_latency(
        udp_sender_func=send_udp_for_test,
        video_capture_func=get_frame_for_test
    )

    if measured_latency is not None:
        print(f"\n✅ Test finished. Latency is approx: {measured_latency:.2f} ms")
    else:
        print("\n❌ Test failed.")
        
    print("Cleaning up resources...")
    cap.release()
    udp_socket.close()
    print("Test complete.")