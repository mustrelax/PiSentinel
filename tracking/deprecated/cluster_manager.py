"""
ClusterManager: Orchestrates cluster lifecycle and target selection.
Identical structure to multi_tracker.py from SOT system.
"""
import math
import numpy as np
from collections import deque

import config
from tracking.track import Track


class ClusterManager:
    """
    Manages cluster formation, lifecycle, and active target selection.
    Implements hysteresis-based switching to prevent oscillation.
    """
    
    # Clustering parameters
    CLUSTER_RADIUS = 180  # Max distance between tracks in same cluster (pixels)
    MIN_CLUSTER_SIZE = 1  # Single drone = valid cluster
    
    # Target selection parameters (STABILITY FOCUSED)
    SWITCH_THRESHOLD = 2.0         # New target must be 100% better (was 1.5)
    SWITCH_COOLDOWN = 20           # 20 frames (~0.67 sec) cooldown (was 15)
    FORCED_LOCK_DURATION = 30      # Minimum 30 frames lock (1 sec) - CANNOT switch
    LOCK_RELEASE_DISTANCE = 0.7    # Release if cluster >70% of frame diagonal away
    
    # Threat scoring weights (identical to SOT)
    WEIGHT_SIZE = 500.0       # Cluster size (DOMINANT)
    WEIGHT_DISTANCE = 100.0   # Distance from center
    WEIGHT_VELOCITY = 15.0    # Speed
    WEIGHT_APPROACH = 50.0    # Approaching center
    WEIGHT_STABILITY = 5.0    # Longer-lived clusters
    WEIGHT_LOCK_BONUS = 600.0 # HUGE bonus for currently locked cluster
    
    def __init__(self):
        self.clusters = []
        self.next_cluster_id = 1
        
        # Import cluster classes
        from tracking.cluster import DroneCluster
        from tracking.cluster_matcher import ClusterMatcher
        
        self.DroneCluster = DroneCluster
        self.matcher = ClusterMatcher(max_distance=0.85)
        
        # Active target management (identical to SOT)
        self.active_cluster_id = None
        self.switch_cooldown_counter = 0
    
    @staticmethod
    def _distance_between_tracks(track1, track2):
        """Calculate Euclidean distance between two tracks"""
        x1, y1 = track1.kf.x[0], track1.kf.x[1]
        x2, y2 = track2.kf.x[0], track2.kf.x[1]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def _form_clusters(self, tracks):
        """
        Form clusters using agglomerative clustering.
        Single-linkage: If any track in cluster A is within CLUSTER_RADIUS
        of any track in cluster B, merge them.
        
        Returns:
            List of track lists (each list = one cluster)
        """
        confirmed_tracks = [t for t in tracks if t.state == Track.CONFIRMED]
        
        if len(confirmed_tracks) == 0:
            return []
        
        # Initialize: each track is its own cluster
        cluster_tracks = [[track] for track in confirmed_tracks]
        
        # Agglomerative merging
        merged = True
        while merged:
            merged = False
            
            for i in range(len(cluster_tracks)):
                for j in range(i + 1, len(cluster_tracks)):
                    # Check if any track in cluster i is close to any in cluster j
                    min_dist = float('inf')
                    
                    for track_i in cluster_tracks[i]:
                        for track_j in cluster_tracks[j]:
                            dist = self._distance_between_tracks(track_i, track_j)
                            if dist < min_dist:
                                min_dist = dist
                    
                    # Merge if within radius
                    if min_dist < self.CLUSTER_RADIUS:
                        cluster_tracks[i].extend(cluster_tracks[j])
                        cluster_tracks.pop(j)
                        merged = True
                        break
                
                if merged:
                    break
        
        # Filter by minimum size
        cluster_tracks = [ct for ct in cluster_tracks if len(ct) >= self.MIN_CLUSTER_SIZE]
        
        return cluster_tracks
    
    def update(self, tracks):
        """
        Main update loop: cluster → predict → match → update → create → delete
        
        Args:
            tracks: List of Track objects from MOT
        
        Returns:
            List of DroneCluster objects
        """
        # 1. FORM CLUSTERS from tracks
        new_cluster_tracks = self._form_clusters(tracks)
        
        # 2. PREDICT all existing clusters
        for cluster in self.clusters:
            cluster.predict()
        
        # 3. MATCH old clusters to new detections (Hungarian)
        if len(self.clusters) > 0 and len(new_cluster_tracks) > 0:
            matches, unmatched_old, unmatched_new = self.matcher.match(
                self.clusters, new_cluster_tracks
            )
            
            # 4. UPDATE matched clusters
            updated_clusters = []
            for old_idx, new_idx in matches:
                old_cluster = self.clusters[old_idx]
                new_tracks = new_cluster_tracks[new_idx]
                
                old_cluster.update(new_tracks)
                updated_clusters.append(old_cluster)
            
            # 5. MARK MISSED old clusters
            for old_idx in unmatched_old:
                self.clusters[old_idx].mark_missed()
                
                # Keep if not deleted
                if not self.clusters[old_idx].is_deleted():
                    updated_clusters.append(self.clusters[old_idx])
                else:
                    print(f"[CLUSTER] C{self.clusters[old_idx].cluster_id} DELETED")
            
            # 6. CREATE new clusters
            for new_idx in unmatched_new:
                new_tracks = new_cluster_tracks[new_idx]
                new_cluster = self.DroneCluster(self.next_cluster_id, new_tracks)
                print(f"[CLUSTER] NEW C{self.next_cluster_id} (size:{new_cluster.size})")
                updated_clusters.append(new_cluster)
                self.next_cluster_id += 1
            
            self.clusters = updated_clusters
        
        elif len(new_cluster_tracks) > 0:
            # No existing clusters → create all
            self.clusters = []
            for new_tracks in new_cluster_tracks:
                new_cluster = self.DroneCluster(self.next_cluster_id, new_tracks)
                print(f"[CLUSTER] INIT C{self.next_cluster_id} (size:{new_cluster.size})")
                self.clusters.append(new_cluster)
                self.next_cluster_id += 1
        
        else:
            # No new detections → all clusters missed
            for cluster in self.clusters:
                cluster.mark_missed()
            
            self.clusters = [c for c in self.clusters if not c.is_deleted()]
        
        return self.clusters
    
    def calculate_threat_score(self, cluster):
        """
        Calculate threat score for a single cluster.
        Identical logic to SOT ThreatAnalyzer.calculate_threat_score()
        
        Higher score = higher threat
        """
        if cluster.state != cluster.CONFIRMED:
            return 0.0
        
        cx, cy = cluster.kf.x[0], cluster.kf.x[1]
        vx, vy = cluster.kf.x[2], cluster.kf.x[3]
        
        # 1. SIZE SCORE (DOMINANT - multiple drones = higher threat)
        size_score = cluster.size * self.WEIGHT_SIZE
        
        # 2. DISTANCE SCORE (inverse - closer = higher threat)
        cx_screen = config.FRAME_WIDTH / 2
        cy_screen = config.FRAME_HEIGHT / 2
        distance = math.sqrt((cx - cx_screen)**2 + (cy - cy_screen)**2)
        max_distance = math.sqrt(cx_screen**2 + cy_screen**2)
        distance_score = self.WEIGHT_DISTANCE * (1.0 - distance / max_distance)
        
        # 3. VELOCITY SCORE (faster = higher threat)
        velocity = math.sqrt(vx**2 + vy**2)
        velocity_score = self.WEIGHT_VELOCITY * velocity
        
        # Bonus for aggressive motion
        if cluster.is_aggressive_motion():
            velocity_score *= 1.5
        
        # 4. APPROACH ANGLE SCORE (heading towards center = higher threat)
        to_center = np.array([cx_screen - cx, cy_screen - cy])
        velocity_vec = np.array([vx, vy])
        
        to_center_norm = np.linalg.norm(to_center)
        velocity_norm = np.linalg.norm(velocity_vec)
        
        if to_center_norm > 1e-6 and velocity_norm > 1e-6:
            cos_angle = np.dot(velocity_vec, to_center) / (velocity_norm * to_center_norm)
            approach_factor = max(0, cos_angle)
        else:
            approach_factor = 0.0
        
        approach_score = self.WEIGHT_APPROACH * approach_factor
        
        # 5. STABILITY SCORE (longer-lived = more reliable)
        stability_score = self.WEIGHT_STABILITY * min(cluster.hits / 10.0, 1.0)
        
        # 6. LOCK BONUS (HUGE - prevents oscillation)
        # If this cluster is currently locked, give massive bonus
        lock_bonus = 0.0
        if hasattr(self, 'active_cluster_id') and cluster.cluster_id == self.active_cluster_id:
            lock_bonus = self.WEIGHT_LOCK_BONUS
        
        # TOTAL THREAT
        total_score = (size_score + distance_score + velocity_score + 
                      approach_score + stability_score + lock_bonus)
        
        return total_score
    
    def select_active_target(self):
        """
        Select the cluster with highest threat score as active target.
        Implements STRICT LOCK PERSISTENCE to prevent oscillation.
        
        Returns:
            (active_cluster, should_switch)
        """
        confirmed_clusters = [c for c in self.clusters if c.state == c.CONFIRMED]
        
        if len(confirmed_clusters) == 0:
            if self.active_cluster_id is not None:
                print(f"[TARGET] Lost C{self.active_cluster_id} - no confirmed clusters")
            self.active_cluster_id = None
            self.switch_cooldown_counter = 0
            return None, False
        
        # Calculate threat scores for all confirmed clusters
        for cluster in confirmed_clusters:
            cluster.threat_score = self.calculate_threat_score(cluster)
        
        # Sort by threat (descending)
        confirmed_clusters.sort(key=lambda c: c.threat_score, reverse=True)
        best_cluster = confirmed_clusters[0]
        
        # INITIAL TARGET SELECTION
        if self.active_cluster_id is None:
            self.active_cluster_id = best_cluster.cluster_id
            self.switch_cooldown_counter = 0
            print(f"[TARGET] SELECT C{best_cluster.cluster_id} (size:{best_cluster.size}, threat:{best_cluster.threat_score:.0f})")
            return best_cluster, True
        
        # Find current active cluster
        current_cluster = next(
            (c for c in confirmed_clusters if c.cluster_id == self.active_cluster_id), 
            None
        )
        
        # EMERGENCY RELEASE: Current cluster lost
        if current_cluster is None or current_cluster.state != current_cluster.CONFIRMED:
            print(f"[TARGET] EMERGENCY: C{self.active_cluster_id} lost → C{best_cluster.cluster_id}")
            self.active_cluster_id = best_cluster.cluster_id
            self.switch_cooldown_counter = 0
            return best_cluster, True
        
        # EMERGENCY RELEASE: Current cluster too far from screen
        cx, cy = current_cluster.get_predicted_center()
        cx_screen = config.FRAME_WIDTH / 2
        cy_screen = config.FRAME_HEIGHT / 2
        distance_from_center = math.sqrt((cx - cx_screen)**2 + (cy - cy_screen)**2)
        max_distance = math.sqrt(cx_screen**2 + cy_screen**2)
        
        if distance_from_center > max_distance * self.LOCK_RELEASE_DISTANCE:
            print(f"[TARGET] EMERGENCY: C{self.active_cluster_id} too far → C{best_cluster.cluster_id}")
            self.active_cluster_id = best_cluster.cluster_id
            self.switch_cooldown_counter = 0
            return best_cluster, True
        
        # FORCED LOCK: Cannot switch for first N frames
        self.switch_cooldown_counter += 1
        if self.switch_cooldown_counter < self.FORCED_LOCK_DURATION:
            # Absolutely cannot switch (prevents oscillation)
            return current_cluster, False
        
        # EMERGENCY SWITCH: Fast-moving threat (after forced lock expires)
        if (best_cluster.is_aggressive_motion() and 
            best_cluster.cluster_id != self.active_cluster_id and
            best_cluster.threat_score > current_cluster.threat_score * 1.3):  # Still requires 30% better
            print(f"[TARGET] EMERGENCY: Fast threat C{best_cluster.cluster_id} (speed:{best_cluster.get_average_speed():.1f})")
            self.active_cluster_id = best_cluster.cluster_id
            self.switch_cooldown_counter = 0
            return best_cluster, True
        
        # COOLDOWN: Wait before allowing normal switch
        if self.switch_cooldown_counter < self.SWITCH_COOLDOWN:
            return current_cluster, False
        
        # HYSTERESIS: Only switch if new target SIGNIFICANTLY better
        # Note: current_cluster already has WEIGHT_LOCK_BONUS added to its threat
        # So best_cluster needs to overcome this bonus
        if best_cluster.threat_score > current_cluster.threat_score * self.SWITCH_THRESHOLD:
            print(f"[TARGET] SWITCH C{self.active_cluster_id} → C{best_cluster.cluster_id} " +
                  f"(threat: {current_cluster.threat_score:.0f} → {best_cluster.threat_score:.0f})")
            self.active_cluster_id = best_cluster.cluster_id
            self.switch_cooldown_counter = 0
            return best_cluster, True
        
        # MAINTAIN current target
        return current_cluster, False
    
    def get_active_cluster(self):
        """Get currently active cluster (or None)"""
        if self.active_cluster_id is None:
            return None
        
        return next(
            (c for c in self.clusters if c.cluster_id == self.active_cluster_id), 
            None
        )