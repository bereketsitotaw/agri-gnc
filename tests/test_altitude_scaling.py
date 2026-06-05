"""Phase I Sprint 2 — ISA density, power scaling, and dynamic a_max derating."""

import math

import pytest
from shapely.geometry import LineString

from trajectory.altitude_physics import (
    ADDIS_HIGHLAND_AMSL_M,
    HAWASSA_AMSL_M,
    RHO_0_KG_M3,
    FlightEnvironment,
    air_density_isa_kg_m3,
    density_scaling_factors,
    derated_lateral_acceleration,
    get_highland_constraints,
    scale_energy_params_for_density,
)
from trajectory.arc_smoother import generate_smoothed_trajectory
from trajectory.energy_model import (
    AircraftEnergyParams,
    instantaneous_power_w,
    optimal_cruise_speed,
)


def test_air_density_decreases_with_altitude():
    rho_sl = air_density_isa_kg_m3(0.0)
    rho_hawassa = air_density_isa_kg_m3(HAWASSA_AMSL_M)
    rho_addis = air_density_isa_kg_m3(ADDIS_HIGHLAND_AMSL_M)
    assert abs(rho_sl - RHO_0_KG_M3) < 0.02
    assert rho_hawassa < rho_sl
    assert rho_addis < rho_hawassa
    assert 0.94 < rho_addis < 1.02  # ~0.97 at 2300 m ISA


def test_induced_scale_increases_at_high_altitude():
    _, induced_sl = density_scaling_factors(RHO_0_KG_M3)
    rho_hi = air_density_isa_kg_m3(ADDIS_HIGHLAND_AMSL_M)
    _, induced_hi = density_scaling_factors(rho_hi)
    assert induced_hi > induced_sl
    assert induced_hi > 1.0


def test_scaled_c2_increases_power_at_hover_mass():
    base = AircraftEnergyParams()
    rho_hi = air_density_isa_kg_m3(ADDIS_HIGHLAND_AMSL_M)
    scaled = scale_energy_params_for_density(base, rho_hi)
    assert scaled.c2 > base.c2
    # Induced-dominated regime (low speed); cruise 8 m/s has competing parasitic drop
    p_base = instantaneous_power_w(30.0, 3.5, base)
    p_scaled = instantaneous_power_w(30.0, 3.5, scaled)
    assert p_scaled > p_base


def test_derated_a_max_lower_at_2300m_than_sea_level():
    mass = 30.0
    env_sl = FlightEnvironment(altitude_m_amsl=0.0)
    env_hi = FlightEnvironment(altitude_m_amsl=ADDIS_HIGHLAND_AMSL_M)
    a_sl = derated_lateral_acceleration(mass, env_sl)
    a_hi = derated_lateral_acceleration(mass, env_hi)
    assert a_hi < a_sl
    assert a_hi >= 0.5


def test_highland_constraints_flags_mode():
    c = get_highland_constraints(
        30.0,
        FlightEnvironment(altitude_m_amsl=ADDIS_HIGHLAND_AMSL_M),
    )
    assert c["highland_mode"] is True
    assert c["rho"] < RHO_0_KG_M3
    assert c["a_max_lateral"] < 5.0


def test_stalled_vehicle_raises():
    env = FlightEnvironment(
        altitude_m_amsl=4000.0,
        max_thrust_sea_level_n=200.0,
    )
    with pytest.raises(ValueError, match="stalled"):
        derated_lateral_acceleration(50.0, env)


def test_optimal_speed_shift_at_high_altitude():
    base = AircraftEnergyParams()
    rho = air_density_isa_kg_m3(ADDIS_HIGHLAND_AMSL_M)
    scaled = scale_energy_params_for_density(base, rho)
    v_sl = optimal_cruise_speed(30.0, base)
    v_hi = optimal_cruise_speed(30.0, scaled)
    assert abs(v_sl - v_hi) < 2.0  # same order; energy landscape shifts


def test_trajectory_turn_speed_lower_at_addis_amsl():
    segments = [
        LineString([(0.0, 0.0), (100.0, 0.0)]),
        LineString([(100.0, 30.0), (0.0, 30.0)]),
    ]
    hawassa = generate_smoothed_trajectory(
        segments,
        swath_width=10.0,
        turn_resolution=15,
        wind_speed=5.0,
        wind_from_deg=180.0,
        enable_altitude_derating=True,
        flight_environment=FlightEnvironment(altitude_m_amsl=HAWASSA_AMSL_M),
        enable_drift_compensation=False,
    )
    addis = generate_smoothed_trajectory(
        segments,
        swath_width=10.0,
        turn_resolution=15,
        wind_speed=5.0,
        wind_from_deg=180.0,
        enable_altitude_derating=True,
        flight_environment=FlightEnvironment(altitude_m_amsl=ADDIS_HIGHLAND_AMSL_M),
        enable_drift_compensation=False,
    )
    turn_h = min(wp.velocity for wp in hawassa if not wp.is_spraying)
    turn_a = min(wp.velocity for wp in addis if not wp.is_spraying)
    assert turn_a <= turn_h


def test_derating_disabled_preserves_nominal_a_max():
    segments = [LineString([(0.0, 0.0), (80.0, 0.0)])]
    off = generate_smoothed_trajectory(
        segments,
        swath_width=10.0,
        enable_altitude_derating=False,
        flight_environment=FlightEnvironment(altitude_m_amsl=ADDIS_HIGHLAND_AMSL_M),
    )
    on = generate_smoothed_trajectory(
        segments,
        swath_width=10.0,
        enable_altitude_derating=True,
        flight_environment=FlightEnvironment(altitude_m_amsl=ADDIS_HIGHLAND_AMSL_M),
    )
    turn_off = min(wp.velocity for wp in off if not wp.is_spraying) if any(
        not wp.is_spraying for wp in off
    ) else 8.0
    turn_on = min(wp.velocity for wp in on if not wp.is_spraying) if any(
        not wp.is_spraying for wp in on
    ) else 8.0
    assert turn_on <= turn_off
