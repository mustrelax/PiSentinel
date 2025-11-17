"""
HUD (Heads-Up Display): Status panel, threat bars, and system information.
"""
import cv2
import config
from tracking.track import Track


def draw_status_panel(frame, active_track, all_tracks):
    """Draw professional HUD status panel in top-left corner"""
    # Panel dimensions
    panel_h = 140
    panel_w = 380
    panel_x = 10
    panel_y = 10
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Border
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 2)
    
    # Header
    cv2.putText(frame, "SYSTEM STATUS", (panel_x + 10, panel_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.line(frame, (panel_x + 10, panel_y + 32), 
            (panel_x + panel_w - 10, panel_y + 32), (100, 100, 100), 1)
    
    # Track statistics
    confirmed = len([t for t in all_tracks if t.state == Track.CONFIRMED])
    fast = len([t for t in all_tracks if t.state == Track.CONFIRMED and t.is_fast_moving()])
    lost = len([t for t in all_tracks if t.state == Track.LOST])
    
    cv2.putText(frame, f"Active Tracks: {confirmed}  Fast: {fast}  Lost: {lost}", 
               (panel_x + 10, panel_y + 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    if active_track:
        _draw_target_info(frame, active_track, panel_x, panel_y)
    else:
        cv2.putText(frame, "No Active Target", (panel_x + 10, panel_y + 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)


def _draw_target_info(frame, track, panel_x, panel_y):
    """Draw primary target information section"""
    # Section header
    cv2.putText(frame, "PRIMARY TARGET", (panel_x + 10, panel_y + 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Calculate metrics
    sx, sy = track.kf.x[0], track.kf.x[1]
    speed = track.get_average_speed()
    
    # Distance estimation
    if track.last_detection:
        x1, y1, x2, y2 = track.last_detection
        pixel_w = max(1, x2 - x1)
        distance = (config.KNOWN_WIDTH_CM * config.FOCAL_LENGTH) / pixel_w
    else:
        distance = 0
    
    # Target info line
    cv2.putText(frame, f"ID: {track.track_id}  Dist: {distance:.0f}cm  Vel: {speed:.1f}px/f", 
               (panel_x + 10, panel_y + 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Threat level bar
    _draw_threat_bar(frame, track, panel_x + 10, panel_y + 110)
    
    # Motion state
    if track.is_aggressive_motion():
        motion_text = "AGGRESSIVE"
        motion_color = (0, 0, 255)
    elif track.is_fast_moving():
        motion_text = "FAST"
        motion_color = (0, 165, 255)
    else:
        motion_text = "NORMAL"
        motion_color = (0, 255, 0)
    
    cv2.putText(frame, f"Motion: {motion_text}", (panel_x + 10, panel_y + 135),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, motion_color, 1)


def _draw_threat_bar(frame, track, bar_x, bar_y):
    """Draw horizontal threat level bar with color gradient"""
    threat_normalized = min(track.threat_score / 300.0, 1.0)
    bar_width = 200
    bar_height = 12
    
    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                 (50, 50, 50), -1)
    
    # Filled bar with color gradient
    filled_width = int(bar_width * threat_normalized)
    if threat_normalized > 0.7:
        bar_color = (0, 0, 255)  # Red - high threat
    elif threat_normalized > 0.4:
        bar_color = (0, 165, 255)  # Orange - medium
    else:
        bar_color = (0, 255, 0)  # Green - low
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height),
                 bar_color, -1)
    
    # Threat percentage text
    cv2.putText(frame, f"Threat: {int(threat_normalized * 100)}%", 
               (bar_x + bar_width + 10, bar_y + 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)