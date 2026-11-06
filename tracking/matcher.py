import math
import numpy as np
from scipy.optimize import linear_sum_assignment

import config


class HungarianMatcher:
    # Optimal data association using Hungarian algorithm.
    # Enhanced for stability.
    
    def __init__(self, max_iou_distance=0.7):  # Stricter than 0.85
        self.max_iou_distance = max_iou_distance
    
    @staticmethod
    def iou(box1, box2):
        #Calculate Intersection over Union
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        
        xi1, yi1 = max(x1, x1_), max(y1, y1_)
        xi2, yi2 = min(x2, x2_), min(y2, y2_)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / max(union_area, 1e-6)
    
    @staticmethod
    def euclidean_distance(box1, box2):
        #Calculate normalized Euclidean distance between centers
        cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
        cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
        
        dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        diagonal = math.sqrt(config.FRAME_WIDTH**2 + config.FRAME_HEIGHT**2)
        return dist / diagonal
    
    @staticmethod
    def size_similarity(box1, box2):
        #Calculate size similarity. 0=same, 1=different.
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        
        area1 = w1 * h1
        area2 = w2 * h2
        
        if area1 <= 0 or area2 <= 0:
            return 1.0
        
        # Ratio of areas (1.0 = same, 2.0 = one is 2x bigger)
        ratio = max(area1, area2) / min(area1, area2)
        
        # Convert to cost (0 = same, 1 = very different)
        return min((ratio - 1.0) / 2.0, 1.0)
    
    def compute_cost_matrix(self, tracks, detections):
        # Hybrid cost matrix with motion-aware weighting.
        num_tracks = len(tracks)
        num_dets = len(detections)
        
        if num_tracks == 0 or num_dets == 0:
            return np.empty((num_tracks, num_dets))
        
        cost_matrix = np.zeros((num_tracks, num_dets))
        
        for i, track in enumerate(tracks):
            pred_box = track.get_predicted_box()
            
            # Get track velocity for prediction
            vx, vy = track.kf.x[2], track.kf.x[3]
            speed = math.sqrt(vx**2 + vy**2)
            
            # Adaptive weighting based on motion and track age
            if track.is_fast_moving():
                # bunu sonra düzelt
                iou_weight = 0.3
                dist_weight = 0.5
                size_weight = 0.2
            elif track.age < 5:
                # New tracks: trust position more
                iou_weight = 0.4
                dist_weight = 0.4
                size_weight = 0.2
            else:
                # Established tracks: balanced
                iou_weight = 0.5
                dist_weight = 0.3
                size_weight = 0.2
            
            for j, det_box in enumerate(detections):
                # 1. IoU cost
                iou_cost = 1.0 - self.iou(pred_box, det_box)
                
                # 2. Distance cost
                dist_cost = self.euclidean_distance(pred_box, det_box)
                
                # 3. Size consistency cost (NEW)
                size_cost = self.size_similarity(pred_box, det_box)
                
                # Combined cost
                base_cost = (iou_weight * iou_cost + 
                            dist_weight * dist_cost + 
                            size_weight * size_cost)
                
                # VELOCITY PREDICTION BONUS
                # If detection is in predicted direction, lower cost
                if speed > 2.0 * config.TARGET_FPS:
                    det_cx = (det_box[0] + det_box[2]) / 2
                    det_cy = (det_box[1] + det_box[3]) / 2
                    pred_cx = (pred_box[0] + pred_box[2]) / 2
                    pred_cy = (pred_box[1] + pred_box[3]) / 2
                    
                    # Vector from predicted to detection
                    dx = det_cx - pred_cx
                    dy = det_cy - pred_cy
                    
                    # Dot product with velocity (normalized)
                    if speed > 0:
                        dot = (dx * vx + dy * vy) / (speed * max(math.sqrt(dx**2 + dy**2), 1))
                        
                        # If detection is in velocity direction, reduce cost
                        if dot > 0.5:
                            base_cost *= 0.9  # 10% bonus
                
                cost_matrix[i, j] = base_cost
        
        return cost_matrix
    
    def match(self, tracks, detections):
        # Perform Hungarian matching with stricter thresholds.
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(tracks))), []
        
        cost_matrix = self.compute_cost_matrix(tracks, detections)
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        for t_idx, d_idx in zip(track_indices, det_indices):
            # Apply adaptive threshold based on track state
            track = tracks[t_idx]
            
            # LOST tracks get more lenient matching (they need to re-acquire)
            if track.state == track.LOST:
                threshold = self.max_iou_distance * 1.2  # 20% more lenient
            elif track.state == track.CONFIRMED and track.age > 10:
                # Established tracks get stricter matching
                threshold = self.max_iou_distance * 0.9
            else:
                threshold = self.max_iou_distance
            
            if cost_matrix[t_idx, d_idx] < threshold:
                matches.append((t_idx, d_idx))
                unmatched_detections.remove(d_idx)
            else:
                unmatched_tracks.append(t_idx)
        
        # Add remaining unmatched tracks
        all_track_indices = set(range(len(tracks)))
        matched_track_indices = set([m[0] for m in matches])
        unmatched_tracks.extend(list(all_track_indices - matched_track_indices))
        
        return matches, unmatched_tracks, unmatched_detections