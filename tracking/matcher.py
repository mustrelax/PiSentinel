"""
HungarianMatcher: Optimal assignment between tracks and detections using
Hungarian algorithm with hybrid IoU + distance cost.
"""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

import config


class HungarianMatcher:
    """
    Performs optimal data association using Hungarian algorithm.
    Cost matrix combines IoU and Euclidean distance with motion-aware weighting.
    """
    
    def __init__(self, max_iou_distance=0.85):
        self.max_iou_distance = max_iou_distance
    
    @staticmethod
    def iou(box1, box2):
        """Calculate Intersection over Union"""
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
        """Calculate normalized Euclidean distance between centers"""
        cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
        cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
        
        dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        diagonal = math.sqrt(config.FRAME_WIDTH**2 + config.FRAME_HEIGHT**2)
        return dist / diagonal
    
    def compute_cost_matrix(self, tracks, detections):
        """
        Compute hybrid cost matrix with motion-aware weighting.
        Fast-moving tracks rely more on position than IoU.
        """
        num_tracks = len(tracks)
        num_dets = len(detections)
        
        if num_tracks == 0 or num_dets == 0:
            return np.empty((num_tracks, num_dets))
        
        cost_matrix = np.zeros((num_tracks, num_dets))
        
        for i, track in enumerate(tracks):
            pred_box = track.get_predicted_box()
            
            # Adaptive weighting based on motion
            if track.is_fast_moving():
                iou_weight = 0.4
                dist_weight = 0.6
            else:
                iou_weight = 0.7
                dist_weight = 0.3
            
            for j, det_box in enumerate(detections):
                iou_cost = 1.0 - self.iou(pred_box, det_box)
                dist_cost = self.euclidean_distance(pred_box, det_box)
                
                cost_matrix[i, j] = iou_weight * iou_cost + dist_weight * dist_cost
        
        return cost_matrix
    
    def match(self, tracks, detections):
        """
        Perform Hungarian matching.
        
        Returns:
            matches: List of (track_idx, detection_idx)
            unmatched_tracks: List of track indices
            unmatched_detections: List of detection indices
        """
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
            if cost_matrix[t_idx, d_idx] < self.max_iou_distance:
                matches.append((t_idx, d_idx))
                unmatched_detections.remove(d_idx)
            else:
                unmatched_tracks.append(t_idx)
        
        all_track_indices = set(range(len(tracks)))
        matched_track_indices = set([m[0] for m in matches])
        unmatched_tracks.extend(list(all_track_indices - matched_track_indices))
        
        return matches, unmatched_tracks, unmatched_detections