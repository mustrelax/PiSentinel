# HUD (Heads-Up Display) module
# Handles visualizations.

import cv2
import numpy as np
from datetime import datetime
import config

def draw_hud(frame, active_track, all_tracks, roi_active=False, inference_ms=0.0, current_sector_name="WIDE", threat_zone=None):
    # Draw minimal HUD.
    h, w = frame.shape[:2]
    primary_color = config.HUD_COLOR_PRIMARY
    
    # 1. Top Bar (Status Line)
    # [PiSentinel] [MODE: AUTO] [INF: 15ms] [LOCK: ON]
    
    bar_height = 40
    # Semi-transparent background for top bar
    sub_img = frame[0:bar_height, 0:w]
    rect = np.full(sub_img.shape, (0, 0, 0), dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1 - config.HUD_OPACITY, rect, config.HUD_OPACITY, 0)
    frame[0:bar_height, 0:w] = res
    
    # System Name (Left)
    cv2.putText(frame, "PiSentinel", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, primary_color, 1)
    
    # Status Indicators
    status_x = 150
    status_y = 25
    
    # ROI Status
    if roi_active:
        roi_text = "ROI: LOCK"
        roi_color = primary_color
    elif config.ZOOM_SEARCH_ENABLED:
        roi_text = f"SCAN: {current_sector_name}"
        roi_color = (0, 200, 255) # Orange-ish for scan
    else:
        roi_text = "ROI: WIDE"
        roi_color = (150, 150, 150)
        
    cv2.putText(frame, f"[{roi_text}]", (status_x, status_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, roi_color, 1)
    
    # Track Count
    confirmed_tracks = [t for t in all_tracks if t.state == t.CONFIRMED]
    confirmed_count = len(confirmed_tracks)
    cv2.putText(frame, f"[TRK: {confirmed_count}]", (status_x + 100, status_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, primary_color, 1)

    # Inference Time and Time (Bottom Right - Minimal)
    time_str = datetime.now().strftime('%H:%M:%S')
    inf_text = f"[INF: {inference_ms:.1f}ms] [{time_str}]"
    
    # Calculate position
    text_size = cv2.getTextSize(inf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = w - text_size[0] - 10
    text_y = h - 10
    
    # Draw background bar for readability
    # burası kalsın
    cv2.rectangle(frame, (text_x - 5, text_y - 20), (w, h), (0, 0, 0), -1)
    
    cv2.putText(frame, inf_text, (text_x, text_y - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, primary_color, 1)
    
    # Lock Status (Right aligned)
    if active_track:
        lock_text = f"LOCK: #{active_track.track_id}"
        lock_color = config.HUD_COLOR_DANGER
        if active_track.state == active_track.LOST:
            lock_text = "LOCK: LOST"
            lock_color = config.HUD_COLOR_WARNING
    else:
        lock_text = "LOCK: SEARCH"
        lock_color = (150, 150, 150)
        
    text_size = cv2.getTextSize(lock_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, lock_text, (w - text_size[0] - 10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, lock_color, 1)
    
    # 2. Crosshair (Center)
    # ---------------------------------------------------------
    cx, cy = w // 2, h // 2
    gap = 10
    length = 20
    # Horizontal
    cv2.line(frame, (cx - gap - length, cy), (cx - gap, cy), primary_color, 1)
    cv2.line(frame, (cx + gap, cy), (cx + gap + length, cy), primary_color, 1)
    # Vertical
    cv2.line(frame, (cx, cy - gap - length), (cx, cy - gap), primary_color, 1)
    cv2.line(frame, (cx, cy + gap), (cx, cy + gap + length), primary_color, 1)
    
    # Center dot
    if active_track:
         cv2.circle(frame, (cx, cy), 2, config.HUD_COLOR_DANGER, -1)
    
    # 3. Target Data Panel (Bottom Left) - Only if locked
    # ---------------------------------------------------------
    if active_track:
        panel_x = 10
        panel_y = h - 100
        
        # Real-world Estimation
        real_dist = active_track.real_pos[2] if hasattr(active_track, 'real_pos') else 0.0
        real_speed = active_track.get_real_speed()
        
        # Threat Color
        t_color = primary_color
        t_str = "SAFE"
        
        if threat_zone is not None:
             # Assuming threat_zone enum values: 0=RED, 1=YELLOW, 2=GREEN from threat_analyzer.py
             # We can just map 0->RED, 1->YELLOW, 2->GREEN
             if threat_zone == 0: # RED
                 t_color = (0, 0, 255)
                 t_str = "CRITICAL"
             elif threat_zone == 1: # YELLOW
                 t_color = (0, 255, 255)
                 t_str = "WARNING"
             else: # GREEN
                 t_color = (0, 255, 0)
                 t_str = "MONITOR"
        
        # Data lines
        # Use 2D pixel speed (reliable) instead of buggy 3D m/s
        pixel_speed = active_track.get_current_speed() if hasattr(active_track, 'get_current_speed') else 0.0
        
        lines = [
            (f"TGT: #{active_track.track_id}", primary_color),
            (f"SCORE: {getattr(active_track, 'threat_score', 0):.0f}", t_color),
            (f"DIST: {real_dist:.1f} m", primary_color),
            (f"SPD: {pixel_speed:.1f} px/f", primary_color),
            (f"AZM: {active_track.system_azimuth:.0f} deg" if hasattr(active_track, 'system_azimuth') else "AZM: N/A", primary_color),
            (f"TRT: {t_str}", t_color),
        ]
        
        for i, (text, color) in enumerate(lines):
            # Smaller font for disclaimer
            font_scale = 0.4 if "ESTIMATED" in text else 0.6
            cv2.putText(frame, text, (panel_x, panel_y + (i * 20)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2 if font_scale > 0.5 else 1)
    
    # 4. Corner Brackets (Decorative)
    # ---------------------------------------------------------
    corner_len = 30
    # Top Left
    cv2.line(frame, (10, 40), (10, 40 + corner_len), primary_color, 1)
    cv2.line(frame, (10, 40), (10 + corner_len, 40), primary_color, 1)
    # Top Right
    cv2.line(frame, (w - 10, 40), (w - 10, 40 + corner_len), primary_color, 1)
    cv2.line(frame, (w - 10, 40), (w - 10 - corner_len, 40), primary_color, 1)
    # Bottom Left
    cv2.line(frame, (10, h - 10), (10, h - 10 - corner_len), primary_color, 1)
    cv2.line(frame, (10, h - 10), (10 + corner_len, h - 10), primary_color, 1)
    # Bottom Right
    cv2.line(frame, (w - 10, h - 10), (w - 10, h - 10 - corner_len), primary_color, 1)
    cv2.line(frame, (w - 10, h - 10), (w - 10 - corner_len, h - 10), primary_color, 1)

def draw_roi_rect(frame, name, roi_size=160):
    # Draw dashed ROI rectangle
    h, w = frame.shape[:2]
    
    # Calculate coords based on name (same logic as controller)
    if name == "CENTER":
        x1, y1 = (w - roi_size)//2, (h - roi_size)//2
    elif name == "TOP-L":
        x1, y1 = 0, 0
    elif name == "TOP-R":
        x1, y1 = w - roi_size, 0
    elif name == "BOT-L":
        x1, y1 = 0, h - roi_size
    else: # BOT-R
        x1, y1 = w - roi_size, h - roi_size
        
    x2, y2 = x1 + roi_size, y1 + roi_size
    
    # Draw dashed rect using external util or simple implementation
    # Simple dashed implementation
    color = (0, 255, 0)
    thickness = 2
    dash_len = 10
    
    # Top/Bottom
    for x in range(x1, x2, dash_len*2):
        cv2.line(frame, (x, y1), (min(x+dash_len, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x+dash_len, x2), y2), color, thickness)
        
    # Left/Right
    for y in range(y1, y2, dash_len*2):
        cv2.line(frame, (x1, y), (x1, min(y+dash_len, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y+dash_len, y2)), color, thickness)
        
    # Label
    cv2.putText(frame, f"SCAN: {name}", (x1+5, y1+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_servo_crosshair(frame, center_x, center_y, dead_zone_radius):
    # Draw center crosshair and deadzone circle
    color = (100, 100, 100)
    cv2.circle(frame, (center_x, center_y), int(dead_zone_radius), color, 1)
    cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), color, 1)
    cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), color, 1)
    cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), color, 1)

def draw_dashed_rect(frame, pt1, pt2, color, thickness=1, dash_length=10):
    # Draw a dashed rectangle
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Top edge
    for x in range(x1, x2, dash_length * 2):
        cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
    # Bottom edge
    for x in range(x1, x2, dash_length * 2):
        cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    # Left edge
    for y in range(y1, y2, dash_length * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
    # Right edge
    for y in range(y1, y2, dash_length * 2):
        cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)