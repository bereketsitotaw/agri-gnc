import math

import numpy as np
from shapely.geometry import LineString

from trajectory.speed_profile import (
    braking_distance,
    safe_cornering_speed,
    wind_angle_from_heading,
)
from trajectory.arc_smoother import generate_smoothed_trajectory


def test_safe_cornering_speed_calm():
    v = safe_cornering_speed(5.0, a_max_lateral=5.0, v_min=0.0, v_max=20.0)
    assert abs(v - math.sqrt(25.0)) < 1e-6


def test_safe_cornering_speed_crosswind_reduces_turn_speed():
    v_calm = safe_cornering_speed(5.0, a_max_lateral=5.0, v_min=0.0, v_max=20.0)
    v_wind = safe_cornering_speed(
        5.0, a_max_lateral=5.0, wind_speed=5.0, wind_angle_rad=math.pi / 2,
        v_min=0.0, v_max=20.0,
    )
    assert v_wind < v_calm


def test_braking_distance_tailwind_increases_distance():
    d_calm = braking_distance(8.0, 5.0, a_brake=2.0)
    d_tail = braking_distance(
        8.0, 5.0, a_brake=2.0, wind_speed=5.0, wind_angle_rad=0.0
    )
    assert d_tail > d_calm


def test_wind_angle_from_heading_pure_crosswind():
    heading = math.pi / 2
    angle, w_along, w_cross = wind_angle_from_heading(heading, 5.0, 90.0)
    assert abs(w_along) < 0.01
    assert abs(abs(w_cross) - 5.0) < 0.01


def test_trajectory_wind_asymmetric_turn_speeds():
    segments = [
        LineString([(0.0, 0.0), (100.0, 0.0)]),
        LineString([(100.0, 100.0), (0.0, 100.0)]),
    ]
    calm = generate_smoothed_trajectory(
        segments, swath_width=10.0, turn_resolution=15, wind_speed=0.0
    )
    windy = generate_smoothed_trajectory(
        segments, swath_width=10.0, turn_resolution=15,
        wind_speed=5.0, wind_from_deg=180.0,
    )
    calm_turn = min(wp.velocity for wp in calm if not wp.is_spraying)
    windy_turn = min(wp.velocity for wp in windy if not wp.is_spraying)
    assert windy_turn <= calm_turn
