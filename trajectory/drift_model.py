"""
Spray drift ballistics for upwind path compensation (Phase I — Sprint 1).

Droplet time-of-fall and cross-track drift drive an ENU upwind offset and crab angle
so the ground footprint aligns with the nominal crop row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trajectory.speed_profile import wind_angle_from_heading


# Lightweight LUT: nozzle characteristic size [µm] → terminal velocity [m/s]
NOZZLE_MICRON_TO_V_TERMINAL: dict[int, float] = {
    150: 0.55,
    200: 0.72,
    250: 0.85,
    300: 1.00,
    400: 1.20,
    500: 1.45,
}


@dataclass(frozen=True)
class SprayDriftParams:
    """Environmental and nozzle parameters for drift lookup."""

    altitude_m: float = 15.0
    v_downwash_m_s: float = 3.5
    nozzle_micron: int = 300
    v_terminal_m_s: float | None = None  # overrides LUT if set

    def terminal_velocity(self) -> float:
        if self.v_terminal_m_s is not None:
            return self.v_terminal_m_s
        return NOZZLE_MICRON_TO_V_TERMINAL.get(
            self.nozzle_micron,
            NOZZLE_MICRON_TO_V_TERMINAL[300],
        )


def fall_time_s(params: SprayDriftParams) -> float:
    """t_fall = h / (v_downwash + v_terminal)."""
    v_sum = params.v_downwash_m_s + params.terminal_velocity()
    if v_sum <= 0:
        raise ValueError("v_downwash + v_terminal must be positive")
    return params.altitude_m / v_sum


def wind_velocity_enu(wind_speed: float, wind_from_deg: float) -> tuple[float, float]:
    """Wind velocity (wx, wy) in ENU (+X East, +Y North); FROM meteorological deg."""
    if wind_speed <= 0:
        return 0.0, 0.0
    from_rad = math.radians(wind_from_deg)
    wx = -wind_speed * math.sin(from_rad)
    wy = -wind_speed * math.cos(from_rad)
    return wx, wy


def drift_displacement_enu(
    wind_speed: float,
    wind_from_deg: float,
    params: SprayDriftParams | None = None,
) -> tuple[float, float]:
    """
    Ground drift vector (dx, dy) [m] from release point if vehicle flies on nominal track.

    d = (wx, wy) * t_fall  (droplet carried with horizontal wind during fall).
    """
    p = params or SprayDriftParams()
    t_fall = fall_time_s(p)
    wx, wy = wind_velocity_enu(wind_speed, wind_from_deg)
    return wx * t_fall, wy * t_fall


def upwind_compensation_offset(
    wind_speed: float,
    wind_from_deg: float,
    params: SprayDriftParams | None = None,
) -> tuple[float, float]:
    """
    Shift commanded path by -drift so footprint lands on the nominal row.

    Returns (delta_x, delta_y) to ADD to nominal waypoints in ENU.
    """
    dx, dy = drift_displacement_enu(wind_speed, wind_from_deg, params)
    return -dx, -dy


def crab_angle_rad(
    path_heading_rad: float,
    wind_speed: float,
    wind_from_deg: float,
    cruise_speed_m_s: float,
) -> float:
    """
    Kinematic crab angle relative to track heading.

    psi_crab = arcsin(w_perp / v_cruise)

    Positive crab rotates vehicle heading counter-clockwise from track heading
    (consistent with w_cross sign from wind_angle_from_heading).
    """
    if cruise_speed_m_s <= 0:
        raise ValueError("cruise_speed must be positive")
    if wind_speed <= 0:
        return 0.0

    _, _, w_cross = wind_angle_from_heading(
        path_heading_rad, wind_speed, wind_from_deg
    )
    ratio = w_cross / cruise_speed_m_s
    if abs(ratio) > 1.0:
        raise ValueError(
            f"Crosswind component |w_perp|={abs(w_cross):.2f} m/s exceeds "
            f"cruise speed {cruise_speed_m_s:.2f} m/s — cannot crab."
        )
    return math.asin(ratio)


def vehicle_yaw_deg_from_track(
    path_heading_rad: float,
    crab_angle_rad_val: float,
) -> float:
    """
    MAVLink yaw: 0 = North, 90 = East (clockwise from North).

    path_heading_rad is math angle from +X East (atan2).
    """
    heading_math_deg = math.degrees(path_heading_rad + crab_angle_rad_val)
    yaw = 90.0 - heading_math_deg
    return yaw % 360.0


def compensate_waypoint(
    x: float,
    y: float,
    path_heading_rad: float,
    cruise_speed_m_s: float,
    wind_speed: float,
    wind_from_deg: float,
    params: SprayDriftParams | None = None,
    crab_reference_speed_m_s: float | None = None,
) -> tuple[float, float, float]:
    """
    Apply upwind offset and return (x', y', yaw_deg).

    cruise_speed_m_s: waypoint velocity setpoint (may be turn/brake speed).
    crab_reference_speed_m_s: speed used for arcsin(w_perp/v) — use leg cruise, not v_turn.
    """
    ox, oy = upwind_compensation_offset(wind_speed, wind_from_deg, params)
    v_crab = crab_reference_speed_m_s if crab_reference_speed_m_s is not None else cruise_speed_m_s
    crab = crab_angle_rad(
        path_heading_rad, wind_speed, wind_from_deg, v_crab
    )
    yaw = vehicle_yaw_deg_from_track(path_heading_rad, crab)
    return x + ox, y + oy, yaw


def footprint_offset_from_nominal(
    nominal_x: float,
    nominal_y: float,
    vehicle_x: float,
    vehicle_y: float,
    wind_speed: float,
    wind_from_deg: float,
    params: SprayDriftParams | None = None,
) -> tuple[float, float]:
    """
    Predicted ground footprint relative to nominal row point [m].

    footprint = vehicle_position + drift_vector
    error = footprint - nominal
    """
    dx, dy = drift_displacement_enu(wind_speed, wind_from_deg, params)
    fx = vehicle_x + dx
    fy = vehicle_y + dy
    return fx - nominal_x, fy - nominal_y
