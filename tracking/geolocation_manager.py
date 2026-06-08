"""Stateful geolocation unit for PiSentinel.

Owns the mapping (own_track_id -> latest WorldEstimate) and a cache of
peer-unit estimates relayed by the server. Controller feeds raw telemetry
in; connector reads outbound estimates out; inbound peer updates from the
socket land in this same place. Future phases add correlation and
triangulation hooks alongside these methods without touching the
controller or the connector.
"""
import time
import logging

from tracking.geolocation import compute_world_position, WorldEstimate

logger = logging.getLogger("GeolocationManager")


class GeolocationManager:
    def __init__(self, system_pos, camera_facing, device_name,
                 k_range=0.25, k_bearing_deg=2.0, min_uncertainty_m=2.0):
        self.system_pos = dict(system_pos)
        self.camera_facing = float(camera_facing)
        self.device_name = device_name
        self.k_range = k_range
        self.k_bearing_deg = k_bearing_deg
        self.min_uncertainty_m = min_uncertainty_m

        self.own_tracks = {}   # track_id -> (WorldEstimate, timestamp)
        self.peer_tracks = {}  # (peer_device, peer_track_id) -> (WorldEstimate, timestamp)

        self.on_own_update = None  # (track_id, WorldEstimate) -> None

    def update_own(self, track_id, pan_angle, tilt_angle,
                   optical_az, optical_elev, range_m, track_state="LOCKED"):
        """Called every telemetry tick for any confirmed track on this unit.

        track_state is 'LOCKED' for the actively engaged primary or 'PASSIVE'
        for any other confirmed track. The server uses it to render engagement
        state on the minimap and to colour C2 alerts.
        """
        if range_m is None or range_m <= 0.0:
            return None

        est = compute_world_position(
            pan_angle, tilt_angle, optical_az, optical_elev, range_m,
            self.system_pos, self.camera_facing,
            k_range=self.k_range,
            k_bearing_deg=self.k_bearing_deg,
            min_uncertainty_m=self.min_uncertainty_m,
        )
        est.track_state = str(track_state or "LOCKED")
        self.own_tracks[track_id] = (est, time.time())

        if self.on_own_update is not None:
            try:
                self.on_own_update(track_id, est)
            except Exception as e:
                logger.warning(f"on_own_update callback failed: {e}")
        return est

    def drop(self, track_id):
        self.own_tracks.pop(track_id, None)

    def handle_peer_update(self, peer_device, peer_track_id, world, uncertainty_m,
                           azimuth_world=None, elevation=None, range_m=None,
                           track_state="LOCKED", timestamp=None):
        """Called by the connector when the server relays another unit's update."""
        if peer_device == self.device_name:
            return  # never treat our own echo as a peer
        est = WorldEstimate(
            x=float(world.get("x", 0.0)),
            y=float(world.get("y", 0.0)),
            z=float(world.get("z", 0.0)),
            uncertainty_m=float(uncertainty_m),
            azimuth_world=float(azimuth_world) if azimuth_world is not None else 0.0,
            elevation=float(elevation) if elevation is not None else 0.0,
            range_m=float(range_m) if range_m is not None else 0.0,
            track_state=str(track_state or "LOCKED"),
        )
        self.peer_tracks[(peer_device, peer_track_id)] = (est, timestamp or time.time())

    def get_own_snapshot(self):
        return {tid: est for tid, (est, _) in self.own_tracks.items()}

    def get_peer_snapshot(self):
        return {key: est for key, (est, _) in self.peer_tracks.items()}

    def prune(self, stale_after_s=5.0):
        """Drop cached estimates older than stale_after_s seconds."""
        now = time.time()
        for tid in list(self.own_tracks.keys()):
            if now - self.own_tracks[tid][1] > stale_after_s:
                self.own_tracks.pop(tid, None)
        for key in list(self.peer_tracks.keys()):
            if now - self.peer_tracks[key][1] > stale_after_s:
                self.peer_tracks.pop(key, None)
