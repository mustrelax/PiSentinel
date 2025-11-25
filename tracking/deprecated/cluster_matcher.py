"""
ClusterMatcher: Hungarian algorithm for optimal cluster matching.
Identical structure to matcher.py from SOT system.
"""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

import config


class ClusterMatcher:
    """
    Performs optimal cluster-to-cluster association using Hungarian algorithm.
    Cost matrix combines:
    1. Center distance (Euclidean)
    2. Velocity similarity (for motion continuity)
    3. Size similarity (cluster size)
    
    Validation Gate: Only consider associations within gate size.
    """
    
    # Validation gate (prevents distant associations)
    VALIDATION_GATE_SIZE = 150  # pixels (normalized will be ~0.15)
    
    def __init__(self, max_distance=0.85):
        self.max_distance = max_distance
    
    @staticmethod
    def euclidean_distance(center1, center2):
        """Calculate normalized Euclidean distance between cluster centers"""
        cx1, cy1 = center1
        cx2, cy2 = center2
        
        dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        
        # Normalize by frame diagonal
        diagonal = math.sqrt(config.FRAME_WIDTH**2 + config.FRAME_HEIGHT**2)
        
        return dist / diagonal
    
    @staticmethod
    def velocity_similarity(vel1, vel2):
        """
        Calculate velocity similarity (0 = identical, 1 = opposite)
        Uses cosine distance for direction + magnitude difference
        """
        vx1, vy1 = vel1
        vx2, vy2 = vel2
        
        mag1 = math.sqrt(vx1**2 + vy1**2)
        mag2 = math.sqrt(vx2**2 + vy2**2)
        
        # Handle zero velocity
        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0  # Low cost if either is stationary
        
        # Cosine similarity for direction
        dot_product = vx1 * vx2 + vy1 * vy2
        cos_sim = dot_product / (mag1 * mag2)
        
        # Convert to distance (0 = same direction, 1 = opposite)
        direction_dist = (1.0 - cos_sim) / 2.0
        
        # Magnitude difference (normalized)
        mag_diff = abs(mag1 - mag2) / max(mag1, mag2)
        
        # Combined velocity cost
        return 0.6 * direction_dist + 0.4 * mag_diff
    
    @staticmethod
    def size_similarity(size1, size2):
        """
        Calculate size similarity (0 = identical, 1 = very different)
        """
        if size1 == 0 or size2 == 0:
            return 1.0
        
        ratio = max(size1, size2) / min(size1, size2)
        
        # Normalized difference (1 = identical, 2 = one is 2x other)
        # Convert to cost (0 = identical, 1 = very different)
        return min((ratio - 1.0) / 3.0, 1.0)  # Cap at 1.0
    
    def compute_cost_matrix(self, old_clusters, new_cluster_tracks):
        """
        Compute hybrid cost matrix with motion-aware weighting.
        
        Args:
            old_clusters: List of DroneCluster from previous frame
            new_cluster_tracks: List of track lists (not yet DroneCluster objects)
        
        Returns:
            cost_matrix: numpy array of shape (num_old, num_new)
        """
        num_old = len(old_clusters)
        num_new = len(new_cluster_tracks)
        
        if num_old == 0 or num_new == 0:
            return np.empty((num_old, num_new))
        
        cost_matrix = np.zeros((num_old, num_new))
        
        for i, old_cluster in enumerate(old_clusters):
            # Get predicted center and velocity from Kalman
            pred_cx, pred_cy = old_cluster.get_predicted_center()
            pred_vx, pred_vy = old_cluster.kf.x[2], old_cluster.kf.x[3]
            old_size = old_cluster.size
            
            # ADAPTIVE WEIGHTING based on motion (like SOT matcher)
            if old_cluster.is_fast_moving():
                # Fast motion → trust position more than size
                center_weight = 0.6
                velocity_weight = 0.25
                size_weight = 0.15
            else:
                # Slow motion → balanced weighting
                center_weight = 0.5
                velocity_weight = 0.2
                size_weight = 0.3
            
            for j, new_tracks in enumerate(new_cluster_tracks):
                # Calculate new cluster center and velocity (simple average)
                new_xs = [t.kf.x[0] for t in new_tracks]
                new_ys = [t.kf.x[1] for t in new_tracks]
                new_cx, new_cy = np.mean(new_xs), np.mean(new_ys)
                
                new_vxs = [t.kf.x[2] for t in new_tracks]
                new_vys = [t.kf.x[3] for t in new_tracks]
                new_vx, new_vy = np.mean(new_vxs), np.mean(new_vys)
                
                new_size = len(new_tracks)
                
                # 1. CENTER DISTANCE COST
                center_cost = self.euclidean_distance(
                    (pred_cx, pred_cy),
                    (new_cx, new_cy)
                )
                
                # 2. VELOCITY SIMILARITY COST
                velocity_cost = self.velocity_similarity(
                    (pred_vx, pred_vy),
                    (new_vx, new_vy)
                )
                
                # 3. SIZE SIMILARITY COST
                size_cost = self.size_similarity(old_size, new_size)
                
                # HYBRID COST (weighted combination)
                total_cost = (center_weight * center_cost + 
                             velocity_weight * velocity_cost + 
                             size_weight * size_cost)
                
                # VALIDATION GATE: Reject if distance too large
                # Check actual pixel distance (before normalization)
                actual_distance = math.sqrt((pred_cx - new_cx)**2 + (pred_cy - new_cy)**2)
                
                if actual_distance > self.VALIDATION_GATE_SIZE:
                    total_cost = 1.0  # Maximum cost (will be rejected)
                
                cost_matrix[i, j] = total_cost
        
        return cost_matrix
    
    def match(self, old_clusters, new_cluster_tracks):
        """
        Perform Hungarian matching between old clusters and new detections.
        
        Args:
            old_clusters: List of DroneCluster objects from previous frame
            new_cluster_tracks: List of track lists (grouped tracks)
        
        Returns:
            matches: List of (old_cluster_idx, new_cluster_idx) tuples
            unmatched_old: List of old cluster indices
            unmatched_new: List of new cluster indices
        """
        if len(old_clusters) == 0:
            return [], [], list(range(len(new_cluster_tracks)))
        
        if len(new_cluster_tracks) == 0:
            return [], list(range(len(old_clusters))), []
        
        # Compute cost matrix
        cost_matrix = self.compute_cost_matrix(old_clusters, new_cluster_tracks)
        
        # Hungarian algorithm
        old_indices, new_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by cost threshold
        matches = []
        unmatched_old = []
        unmatched_new = list(range(len(new_cluster_tracks)))
        
        for old_idx, new_idx in zip(old_indices, new_indices):
            if cost_matrix[old_idx, new_idx] < self.max_distance:
                matches.append((old_idx, new_idx))
                unmatched_new.remove(new_idx)
            else:
                unmatched_old.append(old_idx)
        
        # Add old clusters that weren't assigned
        all_old_indices = set(range(len(old_clusters)))
        matched_old_indices = set([m[0] for m in matches])
        unmatched_old.extend(list(all_old_indices - matched_old_indices))
        
        return matches, unmatched_old, unmatched_new