"""Unit tests for tracking.geolocation.compute_world_position.

Run from repo root:
    python -m pytest tests/test_geolocation.py -v
or standalone:
    python tests/test_geolocation.py
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.geolocation import compute_world_position


ORIGIN = {"x": 0.0, "y": 0.0, "z": 0.0}


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


class TestAzimuthMapping(unittest.TestCase):
    """Spherical -> world-frame azimuth check."""

    def test_camera_forward_facing_north_is_pure_north(self):
        est = compute_world_position(
            pan_angle=90, tilt_angle=90, optical_az=0, optical_elev=0,
            range_m=100, system_pos=ORIGIN, camera_facing=0,
        )
        self.assertTrue(approx(est.x, 0, 1e-4))
        self.assertTrue(approx(est.y, 100, 1e-4))
        self.assertTrue(approx(est.z, 0, 1e-4))
        self.assertTrue(approx(est.azimuth_world, 0, 1e-4))

    def test_camera_forward_facing_east_maps_x(self):
        est = compute_world_position(
            pan_angle=90, tilt_angle=90, optical_az=0, optical_elev=0,
            range_m=100, system_pos=ORIGIN, camera_facing=90,
        )
        self.assertTrue(approx(est.x, 100, 1e-4))
        self.assertTrue(approx(est.y, 0, 1e-4))
        self.assertTrue(approx(est.azimuth_world, 90, 1e-4))

    def test_pan_right_turns_azimuth_right(self):
        # pan=60 -> servo_offset = +30 -> from north, rotate 30 deg east
        est = compute_world_position(
            pan_angle=60, tilt_angle=90, optical_az=0, optical_elev=0,
            range_m=100, system_pos=ORIGIN, camera_facing=0,
        )
        self.assertTrue(approx(est.azimuth_world, 30, 1e-4))
        self.assertTrue(approx(est.x, 100 * math.sin(math.radians(30)), 1e-4))
        self.assertTrue(approx(est.y, 100 * math.cos(math.radians(30)), 1e-4))

    def test_optical_az_adds_on_top_of_servo(self):
        est = compute_world_position(
            pan_angle=90, tilt_angle=90, optical_az=10, optical_elev=0,
            range_m=100, system_pos=ORIGIN, camera_facing=45,
        )
        self.assertTrue(approx(est.azimuth_world, 55, 1e-4))

    def test_azimuth_wraps_mod_360(self):
        est = compute_world_position(
            pan_angle=90, tilt_angle=90, optical_az=20, optical_elev=0,
            range_m=1, system_pos=ORIGIN, camera_facing=350,
        )
        self.assertTrue(approx(est.azimuth_world, 10, 1e-4))


class TestElevationMapping(unittest.TestCase):
    def test_tilt_up_gives_positive_z(self):
        # tilt=60 -> elevation=+30 -> z = r*sin(30) = 50
        est = compute_world_position(
            pan_angle=90, tilt_angle=60, optical_az=0, optical_elev=0,
            range_m=100, system_pos=ORIGIN, camera_facing=0,
        )
        self.assertTrue(approx(est.z, 50, 1e-4))
        self.assertTrue(approx(est.elevation, 30, 1e-4))

    def test_tilt_down_gives_negative_z(self):
        est = compute_world_position(
            pan_angle=90, tilt_angle=120, optical_az=0, optical_elev=0,
            range_m=100, system_pos=ORIGIN, camera_facing=0,
        )
        self.assertTrue(approx(est.z, -50, 1e-4))
        self.assertTrue(approx(est.elevation, -30, 1e-4))

    def test_optical_elev_adds_on_top_of_servo(self):
        est = compute_world_position(
            pan_angle=90, tilt_angle=90, optical_az=0, optical_elev=5,
            range_m=100, system_pos=ORIGIN, camera_facing=0,
        )
        self.assertTrue(approx(est.elevation, 5, 1e-4))

    def test_elevation_clamped(self):
        est = compute_world_position(
            pan_angle=90, tilt_angle=0, optical_az=0, optical_elev=20,
            range_m=100, system_pos=ORIGIN, camera_facing=0,
        )
        self.assertLessEqual(est.elevation, 89.0)


class TestSystemOffset(unittest.TestCase):
    def test_system_position_translates_result(self):
        sys_pos = {"x": 100.0, "y": 200.0, "z": 5.0}
        est = compute_world_position(
            pan_angle=90, tilt_angle=90, optical_az=0, optical_elev=0,
            range_m=50, system_pos=sys_pos, camera_facing=0,
        )
        self.assertTrue(approx(est.x, 100, 1e-4))
        self.assertTrue(approx(est.y, 250, 1e-4))
        self.assertTrue(approx(est.z, 5, 1e-4))


class TestUncertainty(unittest.TestCase):
    def test_uncertainty_grows_with_range(self):
        near = compute_world_position(
            pan_angle=90, tilt_angle=90, optical_az=0, optical_elev=0,
            range_m=50, system_pos=ORIGIN, camera_facing=0,
        )
        far = compute_world_position(
            pan_angle=90, tilt_angle=90, optical_az=0, optical_elev=0,
            range_m=500, system_pos=ORIGIN, camera_facing=0,
        )
        self.assertLess(near.uncertainty_m, far.uncertainty_m)

    def test_min_uncertainty_floor_at_short_range(self):
        est = compute_world_position(
            pan_angle=90, tilt_angle=90, optical_az=0, optical_elev=0,
            range_m=1, system_pos=ORIGIN, camera_facing=0,
            k_range=0.25, k_bearing_deg=2.0, min_uncertainty_m=5.0,
        )
        self.assertGreaterEqual(est.uncertainty_m, 5.0)

    def test_uncertainty_matches_formula(self):
        r = 200.0
        k_r = 0.25
        k_b = 2.0
        est = compute_world_position(
            pan_angle=90, tilt_angle=90, optical_az=0, optical_elev=0,
            range_m=r, system_pos=ORIGIN, camera_facing=0,
            k_range=k_r, k_bearing_deg=k_b, min_uncertainty_m=1.0,
        )
        sigma_r = k_r * r
        sigma_b = math.sin(math.radians(k_b)) * r
        expected = math.sqrt(sigma_r ** 2 + sigma_b ** 2)
        self.assertTrue(approx(est.uncertainty_m, expected, 1e-6))


class TestAsDict(unittest.TestCase):
    def test_as_dict_keys(self):
        est = compute_world_position(
            pan_angle=90, tilt_angle=90, optical_az=0, optical_elev=0,
            range_m=100, system_pos=ORIGIN, camera_facing=0,
        )
        d = est.as_dict()
        self.assertEqual(
            set(d.keys()),
            {"x", "y", "z", "uncertainty_m", "azimuth_world", "elevation", "range_m"},
        )
        for v in d.values():
            self.assertIsInstance(v, float)


if __name__ == "__main__":
    unittest.main()
