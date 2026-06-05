import math

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from trajectory.waypoint import Waypoint
from trajectory.energy_model import (
    AircraftEnergyParams,
    optimal_cruise_speed,
    sweep_mass_schedule,
)
from geometry.spray_geofence import SprayGeofence, SprayLegSlice
from trajectory.altitude_physics import FlightEnvironment, get_highland_constraints
from trajectory.drift_model import SprayDriftParams, compensate_waypoint
from trajectory.speed_profile import (
    braking_distance,
    estimate_min_turn_radius,
    safe_cornering_speed,
    wind_angle_from_heading,
)


def cruise_speeds_for_sweeps(
    segments: list[LineString],
    swath_width: float,
    cruise_speed: float = 8.0,
    use_energy_optimal: bool = False,
    application_rate_l_per_m2: float = 0.02,
    energy_params: AircraftEnergyParams | None = None,
) -> list[float]:
    """Per-sweep cruise speeds: fixed or mass-optimal from the energy model."""
    if not use_energy_optimal:
        return [cruise_speed] * len(segments)
    p = energy_params or AircraftEnergyParams()
    masses = sweep_mass_schedule(
        segments, swath_width, application_rate_l_per_m2, p
    )
    return [optimal_cruise_speed(m, p) for m in masses]


def generate_cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    num_points: int = 10
) -> list[tuple[float, float]]:
    """Generates points along a cubic Bezier curve."""
    t = np.linspace(0, 1, num_points)
    curve = []
    for i in range(num_points):
        b0 = (1 - t[i]) ** 3
        b1 = 3 * t[i] * (1 - t[i]) ** 2
        b2 = 3 * (t[i] ** 2) * (1 - t[i])
        b3 = t[i] ** 3

        x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
        y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
        curve.append((x, y))
    return curve


def _segment_heading(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.atan2(end[1] - start[1], end[0] - start[0])


def _turn_geometry(
    end_coord: tuple[float, float],
    start_coord: tuple[float, float],
    next_seg: LineString,
    swath_width: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Returns p0, p1, p2, p3, unit tangent v1 at sweep exit."""
    p0 = end_coord
    dx1 = end_coord[0] - start_coord[0]
    dy1 = end_coord[1] - start_coord[1]
    len1 = np.hypot(dx1, dy1)
    v1 = (dx1 / len1, dy1 / len1) if len1 > 0 else (0.0, 0.0)

    p3 = next_seg.coords[0]
    p4 = next_seg.coords[-1]
    dx2 = p4[0] - p3[0]
    dy2 = p4[1] - p3[1]
    len2 = np.hypot(dx2, dy2)
    v2 = (dx2 / len2, dy2 / len2) if len2 > 0 else (0.0, 0.0)

    gap_dist = np.hypot(p3[0] - p0[0], p3[1] - p0[1])
    d = max(gap_dist * 0.5, swath_width * 0.5)
    p1 = (p0[0] + v1[0] * d, p0[1] + v1[1] * d)
    p2 = (p3[0] - v2[0] * d, p3[1] - v2[1] * d)
    return p0, p1, p2, p3, v1


def _harsher_wind_angle(
    heading_a: float,
    heading_b: float,
    wind_speed: float,
    wind_from_deg: float,
) -> float:
    """Pick the leg heading with stronger crosswind component (asymmetric turns)."""
    angle_a, _, _ = wind_angle_from_heading(heading_a, wind_speed, wind_from_deg)
    angle_b, _, _ = wind_angle_from_heading(heading_b, wind_speed, wind_from_deg)
    if abs(math.sin(angle_a)) >= abs(math.sin(angle_b)):
        return angle_a
    return angle_b


def _kinematic_turn_speed(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    cruise_speed: float,
    turn_speed: float,
    a_max_lateral: float,
    min_turn_speed: float,
    wind_speed: float,
    wind_angle_rad: float,
) -> float:
    radius = estimate_min_turn_radius(p0, p1, p2, p3)
    v_turn = safe_cornering_speed(
        radius,
        a_max_lateral=a_max_lateral,
        wind_speed=wind_speed,
        wind_angle_rad=wind_angle_rad,
        v_min=min_turn_speed,
        v_max=cruise_speed,
    )
    if turn_speed > 0:
        v_turn = min(v_turn, turn_speed)
    return v_turn


def _apply_spray_drift(
    x: float,
    y: float,
    heading: float,
    waypoint_speed: float,
    wind_speed: float,
    wind_from_deg: float,
    enable: bool,
    drift_params: SprayDriftParams | None,
    leg_cruise_speed: float | None = None,
) -> tuple[float, float, float | None]:
    if not enable or wind_speed <= 0:
        return x, y, None
    p = drift_params or SprayDriftParams(altitude_m=15.0)
    v_crab = leg_cruise_speed if leg_cruise_speed is not None else waypoint_speed
    cx, cy, yaw = compensate_waypoint(
        x, y, heading, waypoint_speed, wind_speed, wind_from_deg, p,
        crab_reference_speed_m_s=v_crab,
    )
    return cx, cy, yaw


def _clip_to_geofence(
    x: float,
    y: float,
    geofence: SprayGeofence | None,
) -> tuple[float, float]:
    if geofence is None:
        return x, y
    pt = Point(x, y)
    if geofence.active_zone.contains(pt):
        return x, y
    on_zone, _ = nearest_points(geofence.active_zone, pt)
    return float(on_zone.x), float(on_zone.y)


def _append_spray_waypoint(
    trajectory: list[Waypoint],
    x: float,
    y: float,
    flight_altitude: float,
    velocity: float,
    heading: float,
    wind_speed: float,
    wind_from_deg: float,
    enable_drift: bool,
    drift_params: SprayDriftParams | None,
    leg_cruise_speed: float,
    geofence: SprayGeofence | None = None,
) -> None:
    sx, sy, syaw = _apply_spray_drift(
        x, y, heading, velocity, wind_speed, wind_from_deg,
        enable_drift, drift_params, leg_cruise_speed=leg_cruise_speed,
    )
    if geofence is not None:
        # Keep commanded path inside enrolled plots; crab yaw may remain for footprint.
        if not geofence.active_zone.contains(Point(sx, sy)):
            sx, sy = x, y
    else:
        sx, sy = _clip_to_geofence(sx, sy, geofence)
    trajectory.append(Waypoint(
        x=sx, y=sy, z=flight_altitude,
        velocity=velocity, is_spraying=True, yaw_deg=syaw,
    ))


def _append_transit_waypoint(
    trajectory: list[Waypoint],
    x: float,
    y: float,
    flight_altitude: float,
    velocity: float,
) -> None:
    trajectory.append(Waypoint(
        x=x, y=y, z=flight_altitude,
        velocity=velocity, is_spraying=False,
    ))


def _append_sliced_leg(
    trajectory: list[Waypoint],
    leg_slice: SprayLegSlice,
    *,
    flight_altitude: float,
    leg_cruise: float,
    end_velocity: float,
    heading: float,
    wind_speed: float,
    wind_from_deg: float,
    enable_drift_compensation: bool,
    drift_params: SprayDriftParams | None,
    brake_insert: tuple[float, float, float] | None,
    end_coord: tuple[float, float],
    geofence: SprayGeofence | None = None,
    tol_m: float = 0.5,
) -> None:
    """Emit waypoints for one geofence slice (spray ON or corridor transit OFF)."""
    if leg_slice.is_spraying:
        _append_spray_waypoint(
            trajectory, leg_slice.start[0], leg_slice.start[1],
            flight_altitude, leg_cruise, heading, wind_speed, wind_from_deg,
            enable_drift_compensation, drift_params, leg_cruise, geofence,
        )
        ends_at_segment_exit = (
            abs(leg_slice.end[0] - end_coord[0]) <= tol_m
            and abs(leg_slice.end[1] - end_coord[1]) <= tol_m
        )
        if brake_insert is not None and ends_at_segment_exit:
            _append_spray_waypoint(
                trajectory, brake_insert[0], brake_insert[1],
                flight_altitude, brake_insert[2], heading, wind_speed, wind_from_deg,
                enable_drift_compensation, drift_params, leg_cruise, geofence,
            )
        _append_spray_waypoint(
            trajectory, leg_slice.end[0], leg_slice.end[1],
            flight_altitude, end_velocity, heading, wind_speed, wind_from_deg,
            enable_drift_compensation, drift_params, leg_cruise, geofence,
        )
    else:
        _append_transit_waypoint(
            trajectory, leg_slice.start[0], leg_slice.start[1],
            flight_altitude, leg_cruise,
        )
        _append_transit_waypoint(
            trajectory, leg_slice.end[0], leg_slice.end[1],
            flight_altitude, leg_cruise,
        )


def _append_continuous_spray_leg(
    trajectory: list[Waypoint],
    start_coord: tuple[float, float],
    end_coord: tuple[float, float],
    *,
    flight_altitude: float,
    leg_cruise: float,
    v_end_sweep: float,
    heading: float,
    wind_speed: float,
    wind_from_deg: float,
    enable_drift_compensation: bool,
    drift_params: SprayDriftParams | None,
    brake_insert: tuple[float, float, float] | None,
    geofence: SprayGeofence | None = None,
) -> None:
    """Legacy single-segment spray leg (no geofence)."""
    _append_spray_waypoint(
        trajectory, start_coord[0], start_coord[1],
        flight_altitude, leg_cruise, heading, wind_speed, wind_from_deg,
        enable_drift_compensation, drift_params, leg_cruise, geofence,
    )
    if brake_insert is not None:
        _append_spray_waypoint(
            trajectory, brake_insert[0], brake_insert[1],
            flight_altitude, brake_insert[2], heading, wind_speed, wind_from_deg,
            enable_drift_compensation, drift_params, leg_cruise, geofence,
        )
    _append_spray_waypoint(
        trajectory, end_coord[0], end_coord[1],
        flight_altitude, v_end_sweep, heading, wind_speed, wind_from_deg,
        enable_drift_compensation, drift_params, leg_cruise, geofence,
    )


def generate_smoothed_trajectory(
    segments: list[LineString],
    swath_width: float,
    flight_altitude: float = 10.0,
    cruise_speed: float = 8.0,
    turn_speed: float = 4.0,
    turn_resolution: int = 15,
    a_max_lateral: float = 5.0,
    a_brake: float = 0.5,
    min_turn_speed: float = 2.0,
    wind_speed: float = 0.0,
    wind_from_deg: float = 180.0,
    use_energy_optimal: bool = False,
    application_rate_l_per_m2: float = 0.02,
    energy_params: AircraftEnergyParams | None = None,
    enable_drift_compensation: bool = False,
    drift_params: SprayDriftParams | None = None,
    flight_environment: FlightEnvironment | None = None,
    enable_altitude_derating: bool = True,
    geofence: SprayGeofence | None = None,
) -> list[Waypoint]:
    """
    Wind-aware boustrophedon trajectory with asymmetric per-leg speed profiles.

    When use_energy_optimal is True, each spray pass uses v* from the mass-varying
    power model (tank depletion estimated from sweep length × swath × app rate).

    wind_from_deg: meteorological FROM direction (Gazebo 0 5 0 m/s → 180°).
    enable_drift_compensation: apply upwind path offset + crab yaw on spray legs.
    drift_params: altitude, v_downwash, nozzle size for fall-time LUT.
    flight_environment: AMSL + temperature for rho(h) and thrust derating.
    enable_altitude_derating: scale power coeffs and reduce a_max at thin air.
    geofence: optional smallholder zones; OFF-corridor slices become transit waypoints.
    """
    if not segments:
        return []

    if drift_params is None and enable_drift_compensation:
        drift_params = SprayDriftParams(altitude_m=flight_altitude)

    p = energy_params or AircraftEnergyParams()
    a_max_lateral_eff = a_max_lateral
    a_brake_eff = a_brake
    energy_params_eff = p

    if enable_altitude_derating:
        env = flight_environment or FlightEnvironment(altitude_m_amsl=flight_altitude)
        max_mass = p.dry_mass_kg + p.payload_mass_kg
        constraints = get_highland_constraints(max_mass, env, p)
        a_max_lateral_eff = constraints["a_max_lateral"]
        energy_params_eff = constraints["energy_params"]
        if a_max_lateral > 0:
            a_brake_eff = a_brake * (a_max_lateral_eff / a_max_lateral)

    sweep_speeds = cruise_speeds_for_sweeps(
        segments,
        swath_width,
        cruise_speed=cruise_speed,
        use_energy_optimal=use_energy_optimal,
        application_rate_l_per_m2=application_rate_l_per_m2,
        energy_params=energy_params_eff,
    )

    trajectory = []

    for i, current_seg in enumerate(segments):
        leg_cruise = sweep_speeds[i]
        start_coord = current_seg.coords[0]
        end_coord = current_seg.coords[-1]
        approach_heading = _segment_heading(start_coord, end_coord)
        approach_wind_angle, _, _ = wind_angle_from_heading(
            approach_heading, wind_speed, wind_from_deg
        )

        dx1 = end_coord[0] - start_coord[0]
        dy1 = end_coord[1] - start_coord[1]
        len1 = np.hypot(dx1, dy1)

        v_end_sweep = leg_cruise
        brake_insert = None
        v_turn = min_turn_speed

        if i < len(segments) - 1:
            next_seg = segments[i + 1]
            p0, p1, p2, p3, v1 = _turn_geometry(
                end_coord, start_coord, next_seg, swath_width
            )
            departure_heading = _segment_heading(next_seg.coords[0], next_seg.coords[-1])
            turn_wind_angle = _harsher_wind_angle(
                approach_heading, departure_heading, wind_speed, wind_from_deg
            )

            v_turn = _kinematic_turn_speed(
                p0, p1, p2, p3, leg_cruise, turn_speed, a_max_lateral_eff, min_turn_speed,
                wind_speed, turn_wind_angle,
            )
            d_brake = braking_distance(
                leg_cruise, v_turn, a_brake_eff,
                wind_speed=wind_speed,
                wind_angle_rad=approach_wind_angle,
            )
            v_end_sweep = v_turn

            if d_brake > 0.1 and len1 > 0.1:
                brake_dist = min(d_brake, max(len1 - 0.5, 0.0))
                if brake_dist > 0.1:
                    brake_insert = (
                        end_coord[0] - v1[0] * brake_dist,
                        end_coord[1] - v1[1] * brake_dist,
                        v_turn,
                    )

        if geofence is not None:
            leg_slices = geofence.slice_spray_leg(start_coord, end_coord)
            for j, leg_slice in enumerate(leg_slices):
                is_last = j == len(leg_slices) - 1
                slice_end_vel = v_end_sweep if is_last else leg_cruise
                slice_brake = brake_insert if (leg_slice.is_spraying and is_last) else None
                _append_sliced_leg(
                    trajectory,
                    leg_slice,
                    flight_altitude=flight_altitude,
                    leg_cruise=leg_cruise,
                    end_velocity=slice_end_vel,
                    heading=approach_heading,
                    wind_speed=wind_speed,
                    wind_from_deg=wind_from_deg,
                    enable_drift_compensation=enable_drift_compensation,
                    drift_params=drift_params,
                    brake_insert=slice_brake,
                    end_coord=end_coord,
                    geofence=geofence,
                )
        else:
            _append_continuous_spray_leg(
                trajectory,
                start_coord,
                end_coord,
                flight_altitude=flight_altitude,
                leg_cruise=leg_cruise,
                v_end_sweep=v_end_sweep,
                heading=approach_heading,
                wind_speed=wind_speed,
                wind_from_deg=wind_from_deg,
                enable_drift_compensation=enable_drift_compensation,
                drift_params=drift_params,
                brake_insert=brake_insert,
                geofence=geofence,
            )

        if i < len(segments) - 1:
            p0, p1, p2, p3, _ = _turn_geometry(
                end_coord, start_coord, segments[i + 1], swath_width
            )
            departure_heading = _segment_heading(
                segments[i + 1].coords[0], segments[i + 1].coords[-1]
            )
            turn_wind_angle = _harsher_wind_angle(
                approach_heading, departure_heading, wind_speed, wind_from_deg
            )
            v_turn = _kinematic_turn_speed(
                p0, p1, p2, p3, leg_cruise, turn_speed, a_max_lateral_eff, min_turn_speed,
                wind_speed, turn_wind_angle,
            )
            curve_points = generate_cubic_bezier(p0, p1, p2, p3, turn_resolution)
            for pt in curve_points[1:-1]:
                trajectory.append(Waypoint(
                    x=pt[0], y=pt[1], z=flight_altitude,
                    velocity=v_turn, is_spraying=False,
                ))

    return trajectory
