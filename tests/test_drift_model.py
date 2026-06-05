"""Phase I Sprint 1 — spray drift compensation and crab angle."""

import math

import numpy as np
import pytest
from shapely.geometry import LineString

from trajectory.arc_smoother import generate_smoothed_trajectory
from trajectory.drift_model import (
    SprayDriftParams,
    crab_angle_rad,
    drift_displacement_enu,
    fall_time_s,
    footprint_offset_from_nominal,
    compensate_waypoint,
    upwind_compensation_offset,
    vehicle_yaw_deg_from_track,
)
from trajectory.speed_profile import wind_angle_from_heading


def test_fall_time_hawassa_defaults():
    p = SprayDriftParams(altitude_m=15.0, v_downwash_m_s=3.5, nozzle_micron=300)
    t = fall_time_s(p)
    assert abs(t - 15.0 / (3.5 + 1.0)) < 1e-6


def test_drift_displacement_scales_with_wind():
    p = SprayDriftParams(altitude_m=15.0)
    calm = drift_displacement_enu(0.0, 180.0, p)
    windy = drift_displacement_enu(5.0, 180.0, p)
    assert calm == (0.0, 0.0)
    assert windy[1] > 0.0  # wind FROM 180 → blows north (+Y)


def test_upwind_offset_opposes_drift():
    p = SprayDriftParams(altitude_m=15.0)
    dx, dy = drift_displacement_enu(5.0, 180.0, p)
    ox, oy = upwind_compensation_offset(5.0, 180.0, p)
    assert abs(ox + dx) < 1e-9
    assert abs(oy + dy) < 1e-9


def test_compensated_footprint_within_one_meter():
    """M1: compensated vehicle track → footprint on nominal row (5 m/s, FROM 180°)."""
    p = SprayDriftParams(altitude_m=15.0, v_downwash_m_s=3.5, nozzle_micron=300)
    nominal = (100.0, 200.0)
    heading = 0.0  # East along row
    cruise = 8.0
    wind_speed = 5.0
    wind_from = 180.0

    vx, vy, _ = compensate_waypoint(
        nominal[0], nominal[1], heading, cruise, wind_speed, wind_from, p
    )
    err_x, err_y = footprint_offset_from_nominal(
        nominal[0], nominal[1], vx, vy, wind_speed, wind_from, p
    )
    err_mag = math.hypot(err_x, err_y)
    assert err_mag <= 1.0


def test_crab_angle_within_asin_bound():
    heading = math.pi / 2  # North
    crab = crab_angle_rad(heading, 5.0, 90.0, 8.0)
    assert abs(crab) <= math.pi / 2
    assert abs(crab) < math.asin(5.0 / 8.0) + 1e-6


def test_crab_rejected_when_crosswind_exceeds_cruise():
    # Track north; wind FROM 90° → full crosswind on leg
    with pytest.raises(ValueError, match="Crosswind"):
        crab_angle_rad(math.pi / 2, 10.0, 90.0, 8.0)


def test_crab_angle_varies_with_heading():
    """Same wind field, different track headings → different crab."""
    wind_speed = 5.0
    wind_from = 180.0
    cruise = 8.0
    crab_east = crab_angle_rad(0.0, wind_speed, wind_from, cruise)
    crab_north = crab_angle_rad(math.pi / 2, wind_speed, wind_from, cruise)
    assert abs(crab_east - crab_north) > 0.01


def test_vehicle_yaw_deg_in_valid_range():
    yaw = vehicle_yaw_deg_from_track(0.0, 0.1)
    assert 0.0 <= yaw < 360.0


def test_nozzle_lut_monotonic_larger_drops_faster_terminal():
    small = SprayDriftParams(nozzle_micron=200).terminal_velocity()
    large = SprayDriftParams(nozzle_micron=500).terminal_velocity()
    assert large > small


def test_trajectory_enables_drift_yaw_on_spray_legs():
    segments = [LineString([(0.0, 0.0), (100.0, 0.0)])]
    base = generate_smoothed_trajectory(
        segments, swath_width=10.0, wind_speed=5.0, wind_from_deg=180.0,
        enable_drift_compensation=False,
    )
    comp = generate_smoothed_trajectory(
        segments, swath_width=10.0, wind_speed=5.0, wind_from_deg=180.0,
        enable_drift_compensation=True,
        drift_params=SprayDriftParams(altitude_m=15.0),
    )
    spray_base = [wp for wp in base if wp.is_spraying]
    spray_comp = [wp for wp in comp if wp.is_spraying]
    assert all(wp.yaw_deg is None for wp in spray_base)
    assert all(wp.yaw_deg is not None for wp in spray_comp)
    # Upwind offset: FROM 180 → drift north → compensate south (lower y)
    assert spray_comp[0].y < spray_base[0].y


def test_trajectory_brake_waypoint_uses_leg_cruise_for_crab():
    """Brake insert at ~2 m/s must not trigger crab arcsin failure under 5 m/s wind."""
    segments = [
        LineString([(0.0, 0.0), (100.0, 0.0)]),
        LineString([(100.0, 30.0), (0.0, 30.0)]),
    ]
    wps = generate_smoothed_trajectory(
        segments,
        swath_width=10.0,
        wind_speed=5.0,
        wind_from_deg=180.0,
        enable_drift_compensation=True,
        drift_params=SprayDriftParams(altitude_m=15.0),
        a_max_lateral=5.0,
        a_brake=0.5,
    )
    assert len(wps) >= 3
    assert all(
        wp.yaw_deg is None or (0 <= wp.yaw_deg < 360)
        for wp in wps if wp.is_spraying
    )


def test_hawassa_reference_wind_footprint_m1():
    """515 m-class leg: 5 m/s FROM 180°, eastbound spray row."""
    p = SprayDriftParams(altitude_m=15.0)
    heading = 0.0
    cruise = 10.0
    nominal_x, nominal_y = 0.0, 0.0
    vx, vy, yaw = compensate_waypoint(
        nominal_x, nominal_y, heading, cruise, 5.0, 180.0, p
    )
    err = math.hypot(*footprint_offset_from_nominal(
        nominal_x, nominal_y, vx, vy, 5.0, 180.0, p
    ))
    assert err < 0.05  # sub-centimeter analytic closure
