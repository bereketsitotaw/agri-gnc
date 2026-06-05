"""Kinematic speed limits for spray turns (lateral accel + braking + wind)."""

import math

import numpy as np


def wind_angle_from_heading(
    path_heading_rad: float,
    wind_speed: float,
    wind_from_deg: float,
) -> tuple[float, float, float]:
    """
    Decompose wind relative to path tangent (ENU: +X East, +Y North).

    wind_from_deg: meteorological direction wind is coming FROM
                   (0=N, 90=E, 180=S, 270=W). Matches Gazebo <wind> 0 5 0 → FROM 180°.

    Returns (wind_angle_rad, w_along, w_cross) where wind_angle_rad is the angle
    between the path tangent and the wind velocity vector.
    """
    if wind_speed <= 0:
        return 0.0, 0.0, 0.0

    from_rad = math.radians(wind_from_deg)
    wx = -wind_speed * math.sin(from_rad)
    wy = -wind_speed * math.cos(from_rad)

    tx = math.cos(path_heading_rad)
    ty = math.sin(path_heading_rad)
    w_along = wx * tx + wy * ty
    w_cross = wx * ty - wy * tx
    wind_angle_rad = math.atan2(w_cross, w_along)
    return wind_angle_rad, w_along, w_cross


def safe_cornering_speed(
    radius: float,
    a_max_lateral: float = 5.0,
    wind_speed: float = 0.0,
    wind_angle_rad: float = 0.0,
    v_min: float = 2.0,
    v_max: float = 8.0,
) -> float:
    """
    Max turn speed given lateral acceleration limit and crosswind degradation.
    wind_angle_rad: angle between path tangent and wind velocity vector.
    """
    if radius <= 0.01:
        return v_min

    w_cross = wind_speed * math.sin(wind_angle_rad)
    budget_sq = (a_max_lateral ** 2) - (w_cross ** 2)
    a_lat_eff = math.sqrt(max(0.1, budget_sq))

    v_turn = math.sqrt(a_lat_eff * radius)
    return float(np.clip(v_turn, v_min, v_max))


def braking_distance(
    v_cruise: float,
    v_turn: float,
    a_brake: float = 0.5,
    wind_speed: float = 0.0,
    wind_angle_rad: float = 0.0,
) -> float:
    """
    Braking distance using ground-relative speeds (tailwind increases closure rate).
    """
    if v_turn >= v_cruise or a_brake <= 0:
        return 0.0

    w_along_track = wind_speed * math.cos(wind_angle_rad)
    v_ground_cruise = max(0.0, v_cruise + w_along_track)
    v_ground_turn = max(0.0, v_turn + w_along_track)

    distance = (v_ground_cruise ** 2 - v_ground_turn ** 2) / (2.0 * a_brake)
    return max(0.0, distance)


def estimate_min_turn_radius(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    samples: int = 50,
) -> float:
    """Minimum radius of curvature along a cubic Bézier (1 / max curvature)."""
    t = np.linspace(0.0, 1.0, samples)
    p0a = np.array(p0)
    p1a = np.array(p1)
    p2a = np.array(p2)
    p3a = np.array(p3)

    max_kappa = 0.0
    for ti in t:
        u = 1.0 - ti
        d1 = (
            3 * u ** 2 * (p1a - p0a)
            + 6 * u * ti * (p2a - p1a)
            + 3 * ti ** 2 * (p3a - p2a)
        )
        d2 = (
            6 * u * (p2a - 2 * p1a + p0a)
            + 6 * ti * (p3a - 2 * p2a + p1a)
        )
        speed = np.linalg.norm(d1)
        if speed < 1e-6:
            continue
        kappa = abs(d1[0] * d2[1] - d1[1] * d2[0]) / (speed ** 3)
        max_kappa = max(max_kappa, kappa)

    if max_kappa < 1e-9:
        return 1e6
    return 1.0 / max_kappa
