import asyncio
import json
import math
import os
from shapely.geometry import Polygon
from geometry.polygon_utils import validate_and_clean_polygon, project_to_utm
from geometry.dry_corridor_demo import hawassa_dry_corridor_geometry
from geometry.spray_geofence import SprayGeofence
from cpp.sweep_optimizer import optimize_sweep_angle
from trajectory.arc_smoother import cruise_speeds_for_sweeps, generate_smoothed_trajectory
from trajectory.altitude_physics import FlightEnvironment, HAWASSA_AMSL_M, get_highland_constraints
from trajectory.drift_model import SprayDriftParams, crab_angle_rad, vehicle_yaw_deg_from_track
from trajectory.energy_model import AircraftEnergyParams
from px4.mavlink_converter import convert_to_mission_items
from px4.mission_uploader import upload_and_execute_mission

# True  = ~12–17 waypoints, fast QGC/Gazebo demo (~2 min)
# False = 152-waypoint full coverage stress test (~10+ min)
DEMO_MODE = True

# Gazebo windy/baylands: <linear_velocity>0 5 0</linear_velocity> → wind FROM 180° (south)
WIND_SPEED = 5.0
WIND_FROM_DEG = 180.0

# Phase 4 benchmark: True = mass-optimal v* per sweep; False = constant 8 m/s baseline
USE_ENERGY_OPTIMAL = True

# Phase I Sprint 1: upwind path offset + crab yaw on spray legs
ENABLE_DRIFT_COMPENSATION = True

# Phase I Sprint 2: rho(h) power scaling + thrust-margin a_max derating
# Override for highland pass: HOME_AMSL_M=2300 PX4_HOME_ALT=2300
HOME_AMSL_M = float(os.environ.get('HOME_AMSL_M', HAWASSA_AMSL_M))
GROUND_TEMPERATURE_C = 15.0
ENABLE_ALTITUDE_DERATING = True

# Phase I Sprint 3: smallholder geofence (dry corridor between Plot A / Plot B)
ENABLE_SPRINT3_GEOFENCE = os.environ.get('ENABLE_SPRINT3_GEOFENCE', '1') == '1'
HAWASSA_HOME_LON = 38.470000
HAWASSA_HOME_LAT = 7.050000


async def main():
    # PX4 home: PX4_HOME_LAT=7.050000, PX4_HOME_LON=38.470000, PX4_HOME_ALT=1708
    anchor = Polygon([
        (HAWASSA_HOME_LON, HAWASSA_HOME_LAT),
        (HAWASSA_HOME_LON + 0.001, HAWASSA_HOME_LAT),
        (HAWASSA_HOME_LON + 0.001, HAWASSA_HOME_LAT + 0.001),
        (HAWASSA_HOME_LON, HAWASSA_HOME_LAT + 0.001),
    ])

    print("--- LAYER 1: Geometry ---")
    _, converter = project_to_utm(validate_and_clean_polygon(anchor))

    sprint3_meta = None
    geofence = None
    if ENABLE_SPRINT3_GEOFENCE:
        poly_utm, geofence, sprint3_meta = hawassa_dry_corridor_geometry(
            converter, HAWASSA_HOME_LON, HAWASSA_HOME_LAT,
        )
        print("Mode: SPRINT 3 dry corridor (Plot A | 20 m gap | Plot B)")
        print(f"  Envelope: 220×200 m UTM | corridor x ∈ "
              f"[{sprint3_meta['corridor_x_utm'][0]:.0f}, {sprint3_meta['corridor_x_utm'][1]:.0f}]")
    else:
        sitl_coords = [
            (38.470000, 7.050000),
            (38.471000, 7.050000),
            (38.471000, 7.051000),
            (38.470000, 7.051000),
        ]
        poly_valid = validate_and_clean_polygon(Polygon(sitl_coords))
        poly_utm, converter = project_to_utm(poly_valid)
        print("Mode: legacy single-field Hawassa patch")

    print(f"UTM Area: {poly_utm.area:.2f} sq meters")

    print("\n--- LAYER 2: CPP & Trajectory ---")
    if DEMO_MODE:
        swath_width = 30.0
        turn_resolution = 5
        print("Mode: DEMO (wide swaths, sparse turns)")
    else:
        swath_width = 10.0
        turn_resolution = 15
        print("Mode: FULL (10 m swaths, dense Bezier turns)")

    print(f"Wind: {WIND_SPEED} m/s FROM {WIND_FROM_DEG}° (0=N, 90=E)")
    print(f"Drift compensation: {'ON' if ENABLE_DRIFT_COMPENSATION else 'OFF'}")
    print(f"Geofence: {'ON' if geofence is not None else 'OFF'}")
    print(f"Altitude derating: {'ON' if ENABLE_ALTITUDE_DERATING else 'OFF'} @ {HOME_AMSL_M:.0f} m AMSL")

    best_angle, segments, metrics = optimize_sweep_angle(
        poly_utm, swath_width, angle_step=5.0
    )
    print(f"Optimized Angle: {best_angle}° | Turns: {metrics['turn_count']} | "
          f"Path length: {metrics['total_length']:.1f}m")

    energy_params = AircraftEnergyParams()
    flight_env = FlightEnvironment(
        altitude_m_amsl=HOME_AMSL_M,
        temperature_c=GROUND_TEMPERATURE_C,
    )
    highland_constraints = None
    if ENABLE_ALTITUDE_DERATING:
        highland_constraints = get_highland_constraints(
            energy_params.dry_mass_kg + energy_params.payload_mass_kg,
            flight_env,
            energy_params,
        )
        print(
            f"  rho={highland_constraints['rho']:.3f} kg/m³ | "
            f"a_max={highland_constraints['a_max_lateral']:.2f} m/s² | "
            f"highland={highland_constraints['highland_mode']}"
        )
    if USE_ENERGY_OPTIMAL:
        sweep_speeds = cruise_speeds_for_sweeps(
            segments,
            swath_width,
            use_energy_optimal=True,
            application_rate_l_per_m2=0.02,
            energy_params=energy_params,
        )
        for idx, v in enumerate(sweep_speeds):
            print(f"  Sweep {idx + 1}: optimal cruise {v:.1f} m/s")
    else:
        print("  Cruise: constant 8.0 m/s (energy optimization OFF)")

    trajectory = generate_smoothed_trajectory(
        segments=segments,
        swath_width=swath_width,
        flight_altitude=15.0,
        cruise_speed=8.0,
        turn_speed=4.0,
        turn_resolution=turn_resolution,
        a_max_lateral=5.0,
        a_brake=0.5,
        min_turn_speed=2.0,
        wind_speed=WIND_SPEED,
        wind_from_deg=WIND_FROM_DEG,
        use_energy_optimal=USE_ENERGY_OPTIMAL,
        application_rate_l_per_m2=0.02,
        energy_params=energy_params,
        enable_drift_compensation=ENABLE_DRIFT_COMPENSATION,
        drift_params=SprayDriftParams(altitude_m=15.0, nozzle_micron=300),
        flight_environment=flight_env,
        enable_altitude_derating=ENABLE_ALTITUDE_DERATING,
        geofence=geofence,
    )
    print(f"Generated {len(trajectory)} smooth waypoints.")
    spray_wps = sum(1 for wp in trajectory if wp.is_spraying)
    transit_wps = sum(1 for wp in trajectory if not wp.is_spraying)
    print(f"  Spray waypoints: {spray_wps} | Transit (corridor/turn): {transit_wps}")

    if sprint3_meta is not None:
        sprint3_meta['planned_waypoints_utm'] = [
            {'x': wp.x, 'y': wp.y, 'is_spraying': wp.is_spraying}
            for wp in trajectory
        ]
        sprint3_meta['enable_geofence'] = True
        os.makedirs('analysis/output', exist_ok=True)
        sprint3_path = 'analysis/output/sprint3_mission_meta.json'
        with open(sprint3_path, 'w', encoding='utf-8') as f:
            json.dump(sprint3_meta, f, indent=2)
        print(f"  Sprint 3 meta → {sprint3_path}")

    # Sidecar for analysis/grade_phase_1.py (Phase I SITL audit)
    if segments:
        c0 = segments[0].coords[0]
        c1 = segments[0].coords[-1]
        leg_heading = math.atan2(c1[1] - c0[1], c1[0] - c0[0])
        cruise_for_grade = sweep_speeds[0] if USE_ENERGY_OPTIMAL else 8.0
        crab = (
            crab_angle_rad(leg_heading, WIND_SPEED, WIND_FROM_DEG, cruise_for_grade)
            if WIND_SPEED > 0 and ENABLE_DRIFT_COMPENSATION
            else 0.0
        )
        meta = {
            'wind_speed_m_s': WIND_SPEED,
            'wind_from_deg': WIND_FROM_DEG,
            'cruise_speed_m_s': cruise_for_grade,
            'path_heading_deg': math.degrees(leg_heading),
            'expected_yaw_deg': vehicle_yaw_deg_from_track(leg_heading, crab),
            'enable_drift_compensation': ENABLE_DRIFT_COMPENSATION,
            'enable_altitude_derating': ENABLE_ALTITUDE_DERATING,
            'home_amsl_m': HOME_AMSL_M,
        }
        if highland_constraints is not None:
            meta['rho_kg_m3'] = highland_constraints['rho']
            meta['a_max_lateral_m_s2'] = highland_constraints['a_max_lateral']
            meta['highland_mode'] = highland_constraints['highland_mode']
        os.makedirs('analysis/output', exist_ok=True)
        meta_path = 'analysis/output/phase1_mission_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        print(f"  Phase I meta → {meta_path}")

    print("\n--- LAYER 3: PX4 Converter ---")
    mission_items = convert_to_mission_items(trajectory, converter)

    # Enforce PX4 255-item limit with a 5-item buffer
    if len(mission_items) > 250:
        raise ValueError(f"Mission has {len(mission_items)} items — exceeds PX4 limit of 255.")
    print(f"Converted {len(mission_items)} MAVSDK mission items (PX4 limit: 250).")

    print("\n--- LAYER 4: SITL Upload ---")
    await upload_and_execute_mission(
        mission_items,
        connection_url="udpin://0.0.0.0:14540",
    )


if __name__ == "__main__":
    asyncio.run(main())
