"""Spherical -> ENU world-frame conversion for PiSentinel tracks.

Pure math. No hardware, no I/O, no tracker state. All inputs are numeric
so this module is trivially unit-testable.

Coordinate convention (local ENU, meters):
    x = East, y = North, z = Up

Angle conventions (all degrees):
    pan_angle    - servo pan, 0..180 (90 = camera-forward, <90 = right, >90 = left)
    tilt_angle   - servo tilt, 15..165 (90 = horizontal, <90 = up, >90 = down)
    optical_az   - target azimuth relative to camera center (right = +)
    optical_elev - target elevation relative to camera horizon (up = +)
    camera_facing - compass bearing of pan=90 (0=N, 90=E, 180=S, 270=W)
"""
import math
from dataclasses import dataclass


@dataclass
class WorldEstimate:
    """World-frame estimate of a tracked drone."""
    x: float              # meters East
    y: float              # meters North
    z: float              # meters Up
    uncertainty_m: float  # 1-sigma circle radius on the ground plane
    azimuth_world: float  # compass bearing to target, degrees 0..360 (0=N, 90=E)
    elevation: float      # elevation angle to target, degrees (+up / -down)
    range_m: float        # slant range from the reporting unit
    track_state: str = "LOCKED"  # 'LOCKED' for the engaged primary, 'PASSIVE' for other confirmed tracks

    def as_dict(self):
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "uncertainty_m": float(self.uncertainty_m),
            "azimuth_world": float(self.azimuth_world),
            "elevation": float(self.elevation),
            "range_m": float(self.range_m),
            "track_state": str(self.track_state),
        }


def compute_world_position(pan_angle, tilt_angle, optical_az, optical_elev,
                           range_m, system_pos, camera_facing,
                           k_range=0.25, k_bearing_deg=2.0, min_uncertainty_m=2.0):
    """Convert a single telemetry tick into a world-frame WorldEstimate.

    system_pos: dict with keys x, y, z (meters, ENU).
    Returns: WorldEstimate.
    """
    servo_offset = 90.0 - pan_angle                       # pan<90 means right (+)
    azimuth_world = (camera_facing + servo_offset + optical_az) % 360.0

    elevation = (90.0 - tilt_angle) + optical_elev        # tilt<90 = up (+)
    elevation = max(-89.0, min(89.0, elevation))

    az_rad = math.radians(azimuth_world)
    el_rad = math.radians(elevation)
    cos_el = math.cos(el_rad)

    x_rel = range_m * cos_el * math.sin(az_rad)           # East
    y_rel = range_m * cos_el * math.cos(az_rad)           # North
    z_rel = range_m * math.sin(el_rad)                    # Up

    x = system_pos["x"] + x_rel
    y = system_pos["y"] + y_rel
    z = system_pos["z"] + z_rel

    sigma_r = max(min_uncertainty_m, k_range * range_m)
    sigma_b = math.sin(math.radians(k_bearing_deg)) * range_m
    uncertainty_m = math.sqrt(sigma_r * sigma_r + sigma_b * sigma_b)

    return WorldEstimate(
        x=x, y=y, z=z,
        uncertainty_m=uncertainty_m,
        azimuth_world=azimuth_world,
        elevation=elevation,
        range_m=range_m,
    )
