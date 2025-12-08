import math
import numpy as np
import logging
import config
from enum import IntEnum
from tracking.track import Track

logger = logging.getLogger(__name__)

class ThreatZone(IntEnum):
    # Threat zones
    RED = 0      # critical
    YELLOW = 1   # moderate
    GREEN = 2    # monitoring


class TacticalScenario(IntEnum):
    #Tactical scenario classification based on confirmed track count
    SINGLE = 0          # 1 target
    SMALL_GROUP = 1     # 2-4 targets
    MEDIUM_SWARM = 2    # 5-10 targets (leader focus)
    LARGE_SWARM = 3     # 10+ targets (leader priority)


class ReAssessmentTimer:
    def __init__(self):
        self.last_assessment_frame = 0
        self.force_assessment = False

    def should_reassess(self, current_frame, current_track, scenario):
        # Should we re-assess targets this frame?
        frames_since_last = current_frame - self.last_assessment_frame

        # Force assessment (manual trigger or emergency)
        if self.force_assessment:
            self.force_assessment = False
            self.last_assessment_frame = current_frame
            return True

        # Determine interval based on threat zone and scenario
        if current_track:
            zone = SwarmAwareThreatAnalyzer.classify_threat_zone(
                current_track,
                config.FRAME_WIDTH / 2,
                config.FRAME_HEIGHT / 2
            )
            interval = SwarmAwareThreatAnalyzer.get_scenario_reassessment_interval(scenario, zone)
        else:
            # No target - quickly scan for new target
            interval = 30  # 1 second

        if frames_since_last >= interval:
            self.last_assessment_frame = current_frame
            return True

        return False

    def force_next_frame(self):
        self.force_assessment = True


class SwarmAwareThreatAnalyzer:
    CLOSE_RANGE_BOX_SIZE = 120   # If bbox > this, drone is close
    SUCCESSOR_SEARCH_RADIUS = 100 # pixels to search for new target at last known pos
    
    @staticmethod
    def count_nearby_drones(track, all_tracks, radius):
        # Count nearby drones (for visualization/info only)
        tx, ty = track.kf.x[0], track.kf.x[1]
        
        nearby_count = 0
        for other_track in all_tracks:
            if other_track.track_id == track.track_id:
                continue
            if other_track.state != Track.CONFIRMED:
                continue
            
            ox, oy = other_track.kf.x[0], other_track.kf.x[1]
            distance = math.sqrt((tx - ox)**2 + (ty - oy)**2)
            
            if distance < radius:
                nearby_count += 1
        
        return nearby_count
    
    @staticmethod
    def is_close_range(track):
        #Check if drone is at close range (large bounding box)
        if track.last_detection is None:
            return False
        
        x1, y1, x2, y2 = track.last_detection
        w, h = x2 - x1, y2 - y1
        
        return max(w, h) > SwarmAwareThreatAnalyzer.CLOSE_RANGE_BOX_SIZE
    
    @staticmethod
    def is_on_screen(track, margin=50):
        # Check if track is within screen bounds
        sx, sy = track.kf.x[0], track.kf.x[1]

        return (sx >= -margin and sx <= config.FRAME_WIDTH + margin and
                sy >= -margin and sy <= config.FRAME_HEIGHT + margin)

    @staticmethod
    def calculate_time_to_impact(track, center_x, center_y):
        # Calculate time until target reaches center (Time-To-Impact mevzusu).
        sx, sy = track.kf.x[0], track.kf.x[1]
        vx, vy = track.kf.x[2], track.kf.x[3]

        # Vector to center
        to_center_x = center_x - sx
        to_center_y = center_y - sy
        distance = math.sqrt(to_center_x**2 + to_center_y**2)

        # Velocity magnitude
        speed = math.sqrt(vx**2 + vy**2)

        # Stationary target
        if speed < 0.5:
            return float('inf')

        # Approach component (dot product projection)
        approach_speed = (vx * to_center_x + vy * to_center_y) / distance

        # Moving away or perpendicular
        if approach_speed <= 0:
            return float('inf')

        # Convert pixels/frame to seconds
        tti_seconds = distance / (approach_speed * config.TARGET_FPS)

        return tti_seconds

    @staticmethod
    def calculate_approach_rate(track, center_x, center_y):
        # Calculate approach speed (px/frame). (+: coming, -: going)
        sx, sy = track.kf.x[0], track.kf.x[1]
        vx, vy = track.kf.x[2], track.kf.x[3]

        # Direction to center (normalized)
        to_center_x = center_x - sx
        to_center_y = center_y - sy
        distance = math.sqrt(to_center_x**2 + to_center_y**2)

        if distance < 1:
            return 0.0

        dir_x = to_center_x / distance
        dir_y = to_center_y / distance

        # Project velocity onto direction
        approach_rate = vx * dir_x + vy * dir_y

        return approach_rate

    @staticmethod
    def predict_trajectory(track, prediction_seconds=3.0):
        # Predict position 3 seconds ahead (ballistic model).
        sx, sy = track.kf.x[0], track.kf.x[1]
        vx, vy = track.kf.x[2], track.kf.x[3]

        # Ballistic prediction
        frames_ahead = prediction_seconds * config.TARGET_FPS
        pred_x = sx + vx * frames_ahead
        pred_y = sy + vy * frames_ahead

        # Confidence (young tracks are more uncertain)
        base_confidence = min(track.hits / 20.0, 1.0)
        time_decay = 1.0 - (prediction_seconds / 10.0)
        confidence = base_confidence * time_decay

        return pred_x, pred_y, confidence

    @staticmethod
    def calculate_acceleration_factor(track):
        # Detect rapid acceleration (buna dikkat et).
        if len(track.speed_history) < 3:
            return 1.0

        recent_speeds = list(track.speed_history)[-3:]
        speed_delta = recent_speeds[-1] - recent_speeds[0]

        # Rapid acceleration (threat increase)
        if speed_delta > config.ACCEL_THRESHOLD_AGGRESSIVE:
            return config.ACCEL_BONUS_FACTOR  # 1.5
        # Deceleration (decreasing threat)
        elif speed_delta < -config.ACCEL_THRESHOLD_AGGRESSIVE:
            return config.DECEL_PENALTY_FACTOR  # 0.8

        return 1.0

    @staticmethod
    def classify_threat_zone(track, center_x, center_y):
        # Classify target by threat zone (RED/YELLOW/GREEN).
        tti = SwarmAwareThreatAnalyzer.calculate_time_to_impact(track, center_x, center_y)
        distance = math.sqrt((track.kf.x[0] - center_x)**2 + (track.kf.x[1] - center_y)**2)
        approach_rate = SwarmAwareThreatAnalyzer.calculate_approach_rate(track, center_x, center_y)

        # RED ZONE (herhangi bir kriter tetiklerse)
        if tti < config.THREAT_ZONE_RED_TTI:
            return ThreatZone.RED
        if distance < config.THREAT_ZONE_RED_DISTANCE:
            return ThreatZone.RED
        if approach_rate > config.THREAT_ZONE_RED_APPROACH:
            return ThreatZone.RED

        # YELLOW ZONE
        if (tti < config.THREAT_ZONE_YELLOW_TTI or
            (config.THREAT_ZONE_RED_DISTANCE <= distance < config.THREAT_ZONE_YELLOW_DISTANCE) or
            (config.THREAT_ZONE_YELLOW_APPROACH <= approach_rate <= config.THREAT_ZONE_RED_APPROACH)):
            return ThreatZone.YELLOW

        # GREEN ZONE
        return ThreatZone.GREEN

    @staticmethod
    def get_adaptive_lock_bonus(threat_zone, lock_age_frames):
        # Bonus based on zone and age. Higher threat = stronger lock.
        # Zone-based base bonus
        if threat_zone == ThreatZone.RED:
            base_bonus = config.LOCK_BONUS_RED       # 200
        elif threat_zone == ThreatZone.YELLOW:
            base_bonus = config.LOCK_BONUS_YELLOW    # 100
        else:  # GREEN
            base_bonus = config.LOCK_BONUS_GREEN     # 50

        # Time decay (30% reduction after 10 seconds)
        decay_factor = 1.0 - min(lock_age_frames / config.LOCK_BONUS_DECAY_FRAMES, 0.3)

        return base_bonus * decay_factor

    def get_switch_threshold(current_zone, candidate_zone):
        # Score diff required to switch. Upgrade=Easy, Downgrade=Hard.
        # Upgrade
        if candidate_zone < current_zone:  # RED=0 < YELLOW=1 < GREEN=2
            return config.SWITCH_THRESHOLD_UPGRADE  # 50

        # Same zone
        if candidate_zone == current_zone:
            if candidate_zone == ThreatZone.RED:
                return config.SWITCH_THRESHOLD_SAME_RED  # 30
            else:
                return config.SWITCH_THRESHOLD_SAME_OTHER  # 80

        # Downgrade
        return config.SWITCH_THRESHOLD_DOWNGRADE  # 150

    @staticmethod
    def detect_scenario(all_tracks):
        # Detect tactical scenario based on confirmed tracks.
        confirmed = [t for t in all_tracks if t.state == Track.CONFIRMED]
        count = len(confirmed)

        if count <= 1:
            return TacticalScenario.SINGLE
        elif count <= config.SCENARIO_SMALL_GROUP_MAX:  # 4
            return TacticalScenario.SMALL_GROUP
        elif count <= config.SCENARIO_MEDIUM_SWARM_MAX:  # 10
            return TacticalScenario.MEDIUM_SWARM
        else:
            return TacticalScenario.LARGE_SWARM

    def detect_swarm_leader(track, all_tracks, swarm_radius=None):
        # Leader score (0-1). Front + Center + Fast.
        if swarm_radius is None:
            swarm_radius = config.SWARM_DETECTION_RADIUS

        confirmed = [t for t in all_tracks if t.state == Track.CONFIRMED]

        if len(confirmed) < config.SWARM_LEADER_MIN_TRACKS:
            return 0.0  # Not a swarm

        # Nearby tracks (swarm members)
        tx, ty = track.kf.x[0], track.kf.x[1]
        nearby = []
        for other in confirmed:
            if other.track_id == track.track_id:
                continue
            ox, oy = other.kf.x[0], other.kf.x[1]
            dist = math.sqrt((tx - ox)**2 + (ty - oy)**2)
            if dist < swarm_radius:
                nearby.append(other)

        if len(nearby) < 2:
            return 0.0  # Not in swarm formation

        center_x, center_y = config.FRAME_WIDTH / 2, config.FRAME_HEIGHT / 2

        # 1. FRONTMOST SCORE (closest to center)
        all_group = nearby + [track]
        distances = [math.sqrt((t.kf.x[0] - center_x)**2 + (t.kf.x[1] - center_y)**2)
                     for t in all_group]
        min_dist = min(distances)
        track_dist = math.sqrt((tx - center_x)**2 + (ty - center_y)**2)
        frontmost_score = 1.0 - min((track_dist - min_dist) / 100.0, 1.0)
        frontmost_score = max(0, frontmost_score)

        # 2. CENTRALITY SCORE (proximity to formation center)
        centroid_x = np.mean([t.kf.x[0] for t in all_group])
        centroid_y = np.mean([t.kf.x[1] for t in all_group])
        dist_to_centroid = math.sqrt((tx - centroid_x)**2 + (ty - centroid_y)**2)
        centrality_score = 1.0 - min(dist_to_centroid / 100.0, 1.0)

        # 3. VELOCITY SCORE (fastest approaching)
        approach_rates = [SwarmAwareThreatAnalyzer.calculate_approach_rate(t, center_x, center_y)
                          for t in all_group]
        max_approach = max(approach_rates) if max(approach_rates) > 0 else 1.0
        track_approach = SwarmAwareThreatAnalyzer.calculate_approach_rate(track, center_x, center_y)
        velocity_score = max(0, min(1, track_approach / max_approach))

        # Weighted combination (frontmost is most important)
        leadership_score = (frontmost_score * 0.5 +
                           centrality_score * 0.2 +
                           velocity_score * 0.3)

        return leadership_score

    def get_scenario_weights(scenario):
        # Optimized weights for scenario
        if scenario == TacticalScenario.SINGLE:
            return {
                'tti': 300.0,
                'distance': 100.0,
                'velocity': 15.0,
                'approach_rate': 80.0,
                'leader': 0.0,        # No swarm
                'confidence': 5.0,
                'trajectory': 50.0
            }

        elif scenario == TacticalScenario.SMALL_GROUP:
            return {
                'tti': 300.0,
                'distance': 80.0,
                'velocity': 20.0,
                'approach_rate': 90.0,
                'leader': 50.0,       # Light leader priority
                'confidence': 5.0,
                'trajectory': 50.0
            }

        elif scenario == TacticalScenario.MEDIUM_SWARM:
            return {
                'tti': 300.0,
                'distance': 60.0,
                'velocity': 25.0,
                'approach_rate': 100.0,
                'leader': 150.0,      # High leader priority
                'confidence': 10.0,
                'trajectory': 50.0
            }

        else:  # LARGE_SWARM
            return {
                'tti': 300.0,
                'distance': 50.0,
                'velocity': 30.0,
                'approach_rate': 110.0,
                'leader': 200.0,      # Maximum leader priority
                'confidence': 10.0,
                'trajectory': 50.0
            }

    def get_scenario_reassessment_interval(scenario, threat_zone):
        # Interval based on scenario/zone. RED=Stable, GREEN=Aggressive.
        base_intervals = {
            TacticalScenario.SINGLE: config.REASSESS_INTERVAL_SINGLE,        # 90 (3 seconds)
            TacticalScenario.SMALL_GROUP: config.REASSESS_INTERVAL_SMALL,   # 60 (2 seconds)
            TacticalScenario.MEDIUM_SWARM: config.REASSESS_INTERVAL_MEDIUM,  # 45 (1.5 seconds)
            TacticalScenario.LARGE_SWARM: config.REASSESS_INTERVAL_LARGE    # 30 (1 second)
        }

        base = base_intervals[scenario]

        # Zone modifier
        if threat_zone == ThreatZone.RED:
            return base + 30  # more stable
        elif threat_zone == ThreatZone.GREEN:
            return max(base - 15, 30)  # more aggressive

        return base


    


    @staticmethod
    def calculate_advanced_threat_score(track, all_tracks, scenario,
                                       is_current_target, lock_age_frames):
        # Comprehensive threat score setup.
        # Components: TTI, Dist, Vel, Appr, Traj, Leader, Conf, LockBonus.
        if track.state not in [Track.CONFIRMED, Track.LOST]:
            return 0.0

        weights = SwarmAwareThreatAnalyzer.get_scenario_weights(scenario)

        center_x = config.FRAME_WIDTH / 2
        center_y = config.FRAME_HEIGHT / 2

        score = 0.0

        # 1. TIME-TO-IMPACT (Highest priority)
        tti = SwarmAwareThreatAnalyzer.calculate_time_to_impact(track, center_x, center_y)
        if tti < 5.0:
            tti_score = weights['tti']  # Critical - full weight
        elif tti < 15.0:
            # Gradual 30-300 points
            tti_score = weights['tti'] * (15.0 - tti) / 10.0
        else:
            tti_score = weights['tti'] * 0.1  # Low threat
        score += tti_score

        # 2. DISTANCE SCORE (Inverse - closer is more dangerous)
        sx, sy = track.kf.x[0], track.kf.x[1]
        distance = math.sqrt((sx - center_x)**2 + (sy - center_y)**2)
        max_distance = math.sqrt(center_x**2 + center_y**2)
        normalized_dist = distance / max_distance
        distance_score = weights['distance'] * (1.0 - normalized_dist)
        score += distance_score

        # 3. VELOCITY SCORE (Fast = high threat)
        speed = track.get_average_speed()
        velocity_score = weights['velocity'] * speed

        # Apply acceleration factor
        accel_factor = SwarmAwareThreatAnalyzer.calculate_acceleration_factor(track)
        velocity_score *= accel_factor

        # Aggressive motion bonus (existing Track feature)
        if track.is_aggressive_motion():
            velocity_score *= 1.3

        score += velocity_score

        # 4. APPROACH RATE
        approach_rate = SwarmAwareThreatAnalyzer.calculate_approach_rate(track, center_x, center_y)
        approach_score = max(0, approach_rate) * weights['approach_rate']
        score += approach_score

        # 5. TRAJECTORY PREDICTION
        pred_x, pred_y, conf = SwarmAwareThreatAnalyzer.predict_trajectory(track, config.TRAJECTORY_PREDICTION_TIME)
        pred_dist = math.sqrt((pred_x - center_x)**2 + (pred_y - center_y)**2)
        if pred_dist < distance:
            # Target will be closer in 3 seconds
            trajectory_score = weights['trajectory'] * conf
            score += trajectory_score

        # 6. SWARM LEADER DETECTION (Scenario dependent)
        if config.SWARM_LEADER_DETECTION_ENABLED and scenario in [TacticalScenario.MEDIUM_SWARM, TacticalScenario.LARGE_SWARM]:
            leadership = SwarmAwareThreatAnalyzer.detect_swarm_leader(track, all_tracks)
            leader_score = leadership * weights['leader']
            score += leader_score

        # 7. CONFIDENCE SCORE
        confidence_factor = min(track.hits / 10.0, 1.0)
        confidence_score = weights['confidence'] * confidence_factor
        score += confidence_score

        # 8. ADAPTIVE LOCK BONUS (Only current target)
        if is_current_target:
            threat_zone = SwarmAwareThreatAnalyzer.classify_threat_zone(track, center_x, center_y)
            lock_bonus = SwarmAwareThreatAnalyzer.get_adaptive_lock_bonus(threat_zone, lock_age_frames)
            score += lock_bonus

        # 9. LOST TRACK PENALTY
        if track.state == Track.LOST:
            score *= 0.7  # 30% penalty

        # Store nearby count for visualization
        track.nearby_drone_count = SwarmAwareThreatAnalyzer.count_nearby_drones(
            track, all_tracks, config.SWARM_DETECTION_RADIUS
        )

        return score

    @staticmethod
    def dynamic_reassessment(current_track, all_tracks, frame_count,
                            reassess_timer, scenario, lock_age_frames):
        # Dynamic target re-assessment (Periodic, Zone-aware, Adaptive).


        # Time to re-assess?
        if not reassess_timer.should_reassess(frame_count, current_track, scenario):
            # Not yet, keep current target
            return current_track, False

        # === RE-ASSESSMENT TRIGGERED ===

        # I'm using debug logging here to avoid spamming the console
        logger.debug(f"[REASSESS] Frame {frame_count}: Evaluating {len(all_tracks)} candidates")

        # Valid tracks (CONFIRMED or LOST)
        valid_tracks = [t for t in all_tracks if t.state in [Track.CONFIRMED, Track.LOST]]

        if not valid_tracks:
            return None, (current_track is not None)

        # Calculate advanced threat scores for all candidates
        scored_tracks = []
        for track in valid_tracks:
            is_current = (track == current_track)

            # Calculate zone
            zone = SwarmAwareThreatAnalyzer.classify_threat_zone(
                track,
                config.FRAME_WIDTH / 2,
                config.FRAME_HEIGHT / 2
            )

            # Calculate score with scenario-specific weights
            score = SwarmAwareThreatAnalyzer.calculate_advanced_threat_score(
                track, all_tracks, scenario,
                is_current, lock_age_frames if is_current else 0
            )

            scored_tracks.append((track, score, zone))

        # Sort by score (descending)
        scored_tracks.sort(key=lambda x: x[1], reverse=True)
        best_track, best_score, best_zone = scored_tracks[0]

        # === INITIAL TARGET SELECTION (no current target) ===
        if current_track is None:
            # I'm using debug logging here to avoid spamming the console
            logger.debug(f"[REASSESS] Initial lock: #{best_track.track_id} "
                  f"(zone:{best_zone.name}, score:{best_score:.0f})")
            return best_track, True

        # === FIND CURRENT TARGET IN LIST ===
        current_entry = next((e for e in scored_tracks if e[0] == current_track), None)

        if current_entry is None:
            # Current target lost/deleted (EMERGENCY)
            logger.info(f"[REASSESS] EMERGENCY: Target lost, switching to #{best_track.track_id}")
            reassess_timer.force_next_frame()  # Check again immediately
            return best_track, True

        current_score, current_zone = current_entry[1], current_entry[2]

        # === EVALUATE SWITCH DECISION ===
        score_diff = best_score - current_score
        threshold = SwarmAwareThreatAnalyzer.get_switch_threshold(current_zone, best_zone)

        # Decision
        if score_diff > threshold:
            # SWITCH
            logger.info(f"[REASSESS] SWITCH: "
                  f"#{current_track.track_id} (zone:{current_zone.name}, score:{current_score:.0f}) -> "
                  f"#{best_track.track_id} (zone:{best_zone.name}, score:{best_score:.0f}) "
                  f"[diff:{score_diff:.0f} > thresh:{threshold:.0f}]")
            return best_track, True
        else:
            # MAINTAIN LOCK
            if score_diff > threshold * 0.7:
                # Close decision - log for analysis
                logger.info(f"[REASSESS] MAINTAIN: #{current_track.track_id} "
                      f"(diff:{score_diff:.0f} < thresh:{threshold:.0f})")
            return current_track, False

    @staticmethod
    def get_lock_status(current_track, all_tracks):
        # Get detailed lock status for HUD.
        if current_track is None:
            return {
                'locked': False,
                'target_id': None,
                'nearby_count': 0,
                'is_close_range': False,
                'ignored_count': 0
            }
        
        nearby = current_track.nearby_drone_count if hasattr(current_track, 'nearby_drone_count') else 0
        
        # Count how many tracks we're ignoring
        ignored = len([t for t in all_tracks 
                      if t.state == Track.CONFIRMED and t.track_id != current_track.track_id])
        
        return {
            'locked': True,
            'target_id': current_track.track_id,
            'nearby_count': nearby,
            'is_close_range': SwarmAwareThreatAnalyzer.is_close_range(current_track),
            'ignored_count': ignored
        }