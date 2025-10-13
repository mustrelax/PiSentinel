# calibrate_servos.py (V2 - Direct Angle Input)

import socket
import time

# Import servo libraries
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

import config # Import limits from our config file

# --- UDP Sender Setup (to send commands to Unreal) ---
UDP_IP = config.SIM_PC_IP
UDP_PORT = config.SIM_CMD_PORT
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"Connecting to Unreal Engine at {UDP_IP}:{UDP_PORT}")
# --- ---

# --- PCA9685 Servo HAT Setup ---
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c, address=0x40)
pca.frequency = 50
pan_servo = servo.Servo(pca.channels[0])
tilt_servo = servo.Servo(pca.channels[1])
# --- ---

# --- Go to initial position ---
pan_angle = config.PAN_START
tilt_angle = config.TILT_START
pan_servo.angle = pan_angle
tilt_servo.angle = tilt_angle
message = f"{pan_angle},{tilt_angle}"
sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
# --- ---

print("\n--- Direct Angle Calibration Mode ---")
print("Enter Pan and Tilt angles separated by a space (e.g., 90 45)")
print("Type 'q' to quit.")

try:
    while True:
        # Get command from user input
        command = input(f"Enter Pan ({config.PAN_MIN}-{config.PAN_MAX}) & Tilt ({config.TILT_MIN}-{config.TILT_MAX}) > ").lower()

        if command == 'q':
            print("Quitting...")
            break
        
        try:
            # Split the input string by space and convert parts to float
            pan_str, tilt_str = command.split()
            pan_angle = float(pan_str)
            tilt_angle = float(tilt_str)

            # Clamp the angles to the physical limits
            pan_angle = max(config.PAN_MIN, min(config.PAN_MAX, pan_angle))
            tilt_angle = max(config.TILT_MIN, min(config.TILT_MAX, tilt_angle))

            # --- SEND COMMANDS TO BOTH WORLDS ---
            # 1. Send to physical servos
            pan_servo.angle = pan_angle
            tilt_servo.angle = tilt_angle

            # 2. Send to virtual camera in Unreal
            # For now, we send the same values. We will modify this based on observation.
            sim_pan = 90 - pan_angle
            sim_tilt = 90 - tilt_angle

            message = f"{sim_pan},{sim_tilt}"
            sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
            # --- ---
            print(f"--> Sent to Physical: Pan={pan_angle:.1f}, Tilt={tilt_angle:.1f} | Sent to Sim: Pan={sim_pan:.1f}, Tilt={sim_tilt:.1f}")

        except ValueError:
            print("Invalid input. Please enter two numbers separated by a space (e.g., '100 80').")
        
finally:
    # Clean up
    print("\nCentering servos...")
    pan_servo.angle = config.PAN_START
    tilt_servo.angle = config.TILT_START
    message = f"{config.PAN_START},{config.TILT_START}"
    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
    time.sleep(0.5)
    pca.deinit()
    print("Calibration finished.")