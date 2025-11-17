"""
Drawing utilities: Crosshairs, boxes, trajectories, and overlays.
"""
import cv2
import numpy as np
from tracking.track import Track


def draw_crosshair(frame, cx, cy, size=15, color=(0, 0, 255)):
    """Draw professional crosshair on target"""
    # Horizontal line
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 2)
    # Vertical line
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 2)
    # Center dot
    cv2.circle(frame, (cx, cy), 3, color, -1)


def draw_dashed_box(frame, x1, y1, x2, y2, color, thickness=2, dash_length=10):
    """Draw dashed rectangle for tentative/lost tracks"""
    # Top
    for x in range(x1, x2, dash_length * 2):
        cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
    # Bottom
    for x in range(x1, x2, dash_length * 2):
        cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    # Left
    for y in range(y1, y2, dash_length * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
    # Right
    for y in range(y1, y2, dash_length * 2):
        cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)


def draw_track_box(frame, track, is_active):
    """Draw box and label for a single track"""
    box = track.get_predicted_box()
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Determine style based on state
    if is_active:
        color = (0, 0, 255)  # RED
        thickness = 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        draw_crosshair(frame, cx, cy, size=20, color=color)
        cv2.putText(frame, f"TGT-{track.track_id}", (x1, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    elif track.state == Track.CONFIRMED:
        if track.is_aggressive_motion():
            color = (0, 140, 255)  # ORANGE
        elif track.is_fast_moving():
            color = (0, 200, 255)  # LIGHT ORANGE
        else:
            color = (0, 255, 255)  # YELLOW
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f"ID-{track.track_id}", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
    elif track.state == Track.LOST:
        color = (100, 100, 100)  # GRAY
        thickness = 1
        draw_dashed_box(frame, x1, y1, x2, y2, color, thickness)
        cv2.putText(frame, f"LOST-{track.track_id}", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
    else:  # TENTATIVE
        color = (0, 255, 0)  # GREEN
        thickness = 1
        draw_dashed_box(frame, x1, y1, x2, y2, color, thickness)
        cv2.putText(frame, f"NEW-{track.track_id}", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return color


def draw_trajectory(frame, track, color):
    """Draw movement trajectory line"""
    if len(track.trajectory) > 2 and track.state == Track.CONFIRMED:
        points = np.array(track.trajectory, dtype=np.int32)
        cv2.polylines(frame, [points], False, color, 1, cv2.LINE_AA)


def draw_velocity_vector(frame, track, color):
    """Draw velocity arrow for moving tracks"""
    cx, cy = int(track.kf.x[0]), int(track.kf.x[1])
    vx, vy = track.kf.x[2], track.kf.x[3]
    
    if abs(vx) > 1 or abs(vy) > 1:
        end_x = int(cx + vx * 4)
        end_y = int(cy + vy * 4)
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y),
                       color, 1, tipLength=0.3, line_type=cv2.LINE_AA)


def draw_center_reticle(frame, center_x, center_y, dead_zone):
    """Draw center crosshair and dead-zone circle"""
    # Dead-zone circle
    cv2.circle(frame, (center_x, center_y), int(dead_zone), (100, 100, 100), 1, cv2.LINE_AA)
    
    # Center crosshair
    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (100, 100, 100), 1)
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (100, 100, 100), 1)


def draw_tracking_line(frame, center_x, center_y, target_x, target_y):
    """Draw line from center to active target"""
    cv2.line(frame, (center_x, center_y), (int(target_x), int(target_y)),
            (0, 0, 255), 1, cv2.LINE_AA)