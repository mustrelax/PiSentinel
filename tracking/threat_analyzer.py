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
    def get_min_lock_duration_frames(scenario):
        if scenario == TacticalScenario.SINGLE:
            return config.MIN_LOCK_DURATION_SINGLE
        if scenario == TacticalScenario.SMALL_GROUP:
            return config.MIN_LOCK_DURATION_SMALL
        if scenario == TacticalScenario.MEDIUM_SWARM:
            return config.MIN_LOCK_DURATION_MEDIUM
        return config.MIN_LOCK_DURATION_LARGE

    @staticmethod
    def calculate_local_density_score(track, all_tracks, radius):
        tx, ty = track.kf.x[0], track.kf.x[1]
        nearby = 0
        for other in all_tracks:
            if other.track_id == track.track_id or other.state != Track.CONFIRMED:
                continue
            ox, oy = other.kf.x[0], other.kf.x[1]
            if math.sqrt((tx - ox) ** 2 + (ty - oy) ** 2) < radius:
                nearby += 1
        # Saturate quickly so huge swarms do not dominate everything.
        return min(nearby / 4.0, 1.0)
    
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
        # Normalized weights — all components 0-1, weights sum to 1.0.
        # Score = sum(weight * norm_component) * 1000  → range 0-1000.
        # Priority: proximity + approach dominant, density grows with swarm size.
        if scenario == TacticalScenario.SINGLE:
            return {
                'proximity':   0.35,
                'approach':    0.33,
                'tti':         0.18,
                'velocity':    0.11,
                'density':     0.00,  # No swarm context
                'confidence':  0.03,
            }

        elif scenario == TacticalScenario.SMALL_GROUP:
            return {
                'proximity':   0.30,
                'approach':    0.31,
                'tti':         0.17,
                'velocity':    0.10,
                'density':     0.09,  # Cluster bonus starts here
                'confidence':  0.03,
            }

        elif scenario == TacticalScenario.MEDIUM_SWARM:
            return {
                'proximity':   0.27,
                'approach':    0.28,
                'tti':         0.14,
                'velocity':    0.09,
                'density':     0.19,  # Significant cluster bonus
                'confidence':  0.03,
            }

        else:  # LARGE_SWARM
            return {
                'proximity':   0.23,
                'approach':    0.25,
                'tti':         0.13,
                'velocity':    0.09,
                'density':     0.27,  # Dominant cluster bonus
                'confidence':  0.03,
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
        # Normalized threat score: all components 0-1, scaled to 0-1000.
        # Priority order: proximity > approach speed > TTI > velocity > density > confidence.
        if track.state not in [Track.CONFIRMED, Track.LOST]:
            return 0.0

        weights = SwarmAwareThreatAnalyzer.get_scenario_weights(scenario)

        center_x = config.FRAME_WIDTH / 2
        center_y = config.FRAME_HEIGHT / 2
        sx, sy = track.kf.x[0], track.kf.x[1]

        # --- 1. PROXIMITY (linear falloff from center to corner) ---
        distance = math.sqrt((sx - center_x)**2 + (sy - center_y)**2)
        max_distance = math.sqrt(center_x**2 + center_y**2)  # screen corner distance
        proximity_norm = max(0.0, 1.0 - distance / max_distance)

        # --- 2. APPROACH SPEED (positive = toward center) ---
        approach_rate = SwarmAwareThreatAnalyzer.calculate_approach_rate(track, center_x, center_y)
        approach_norm = max(0.0, min(approach_rate / config.APPROACH_RATE_MAX, 1.0))

        # Trajectory bonus: if drone predicted closer in Ns, boost approach by 10%
        pred_x, pred_y, pred_conf = SwarmAwareThreatAnalyzer.predict_trajectory(
            track, config.TRAJECTORY_PREDICTION_TIME)
        pred_dist = math.sqrt((pred_x - center_x)**2 + (pred_y - center_y)**2)
        if pred_dist < distance and pred_conf > 0.3:
            approach_norm = min(approach_norm * 1.1, 1.0)

        # --- 3. TIME-TO-IMPACT (0s=1.0, YELLOW_TTI+=0.0) ---
        tti = SwarmAwareThreatAnalyzer.calculate_time_to_impact(track, center_x, center_y)
        tti_norm = max(0.0, min(1.0 - tti / config.THREAT_ZONE_YELLOW_TTI, 1.0))

        # --- 4. VELOCITY (overall speed, not just approach) ---
        speed = track.get_average_speed()
        velocity_norm = min(speed / config.VELOCITY_NORM_MAX, 1.0)
        # Acceleration factor nudges velocity component slightly (1.0–1.3x)
        accel = SwarmAwareThreatAnalyzer.calculate_acceleration_factor(track)
        velocity_norm = min(velocity_norm * accel, 1.0)

        # --- 5. SWARM DENSITY (0-1, saturates at 4 neighbors) ---
        density_norm = SwarmAwareThreatAnalyzer.calculate_local_density_score(
            track, all_tracks, config.SWARM_DENSITY_RADIUS
        )

        # --- 6. CONFIDENCE ---
        confidence_norm = min(track.hits / 10.0, 1.0)

        # --- Weighted sum scaled to 0-1000 ---
        base_score = (
            weights['proximity']   * proximity_norm +
            weights['approach']    * approach_norm +
            weights['tti']         * tti_norm +
            weights['velocity']    * velocity_norm +
            weights['density']     * density_norm +
            weights['confidence']  * confidence_norm
        ) * 1000.0

        # --- ADAPTIVE LOCK BONUS (only current target, in 0-1000 scale) ---
        if is_current_target:
            threat_zone = SwarmAwareThreatAnalyzer.classify_threat_zone(track, center_x, center_y)
            lock_bonus = SwarmAwareThreatAnalyzer.get_adaptive_lock_bonus(threat_zone, lock_age_frames)
            base_score += lock_bonus

        # --- LOST TRACK PENALTY ---
        if track.state == Track.LOST:
            base_score *= 0.7

        # Store nearby count for visualization
        track.nearby_drone_count = SwarmAwareThreatAnalyzer.count_nearby_drones(
            track, all_tracks, config.SWARM_DETECTION_RADIUS
        )

        return base_score

    @staticmethod
    def dynamic_reassessment(current_track, all_tracks, frame_count,
                            reassess_timer, scenario, lock_age_frames):
        # Dynamic target re-assessment (Periodic, Zone-aware, Adaptive).

        # Post-switch cooldown: don't reassess for 2 seconds after a switch
        MIN_LOCK_DURATION = SwarmAwareThreatAnalyzer.get_min_lock_duration_frames(scenario)
        if current_track and lock_age_frames < MIN_LOCK_DURATION:
            return current_track, False

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
            raw_score = SwarmAwareThreatAnalyzer.calculate_advanced_threat_score(
                track, all_tracks, scenario,
                is_current, lock_age_frames if is_current else 0
            )
            prev = getattr(track, 'stable_threat_score', raw_score)
            # Asymmetric EMA: fast rise (threat escalation), slow decay (sticky high)
            alpha = config.THREAT_SCORE_RISE_ALPHA if raw_score >= prev else config.THREAT_SCORE_DECAY_ALPHA
            stable_score = (alpha * raw_score) + ((1.0 - alpha) * prev)
            track.stable_threat_score = stable_score

            scored_tracks.append((track, stable_score, zone, raw_score))

        # Sort by score (descending)
        scored_tracks.sort(key=lambda x: x[1], reverse=True)
        best_track, best_score, best_zone, best_raw_score = scored_tracks[0]

        # Log threat scores (only when multiple tracks or periodically)
        should_log = len(scored_tracks) > 1 or (frame_count % 90 == 0)
        if should_log and len(scored_tracks) >= 1:
            center_x, center_y = config.FRAME_WIDTH / 2, config.FRAME_HEIGHT / 2
            log_parts = []
            for t, s, z, raw in scored_tracks[:4]:  # Max 4 tracks to avoid spam
                tti = SwarmAwareThreatAnalyzer.calculate_time_to_impact(t, center_x, center_y)
                approach = SwarmAwareThreatAnalyzer.calculate_approach_rate(t, center_x, center_y)
                
                # TTI display with direction
                if tti < 999:
                    tti_str = f"{tti:.1f}s"
                elif approach < -2:
                    tti_str = "RECEDE"  # Moving away
                elif abs(approach) < 2:
                    tti_str = "LATERAL"  # Moving sideways
                else:
                    tti_str = "STATIC"
                    
                marker = "*" if t == current_track else ""
                log_parts.append(f"#{t.track_id}{marker}:{z.name[0]}|{s:.0f}({raw:.0f})|{tti_str}")
            logger.info(f"[THREAT] {' | '.join(log_parts)}")

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
        threshold *= config.SWITCH_MARGIN_MULTIPLIER

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