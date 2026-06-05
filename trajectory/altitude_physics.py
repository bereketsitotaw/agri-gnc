"""
High-altitude air density and thrust-margin derating (Phase I — Sprint 2).

ISA troposphere density, power-coefficient scaling vs. rho, and dynamic a_max for
kinematic planning in Ethiopian highland envelopes (>= 2,300 m AMSL).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trajectory.energy_model import AircraftEnergyParams

# Sea-level standard atmosphere
RHO_0_KG_M3 = 1.225
P_0_PA = 101_325.0
T_0_K = 288.15
LAPSE_RATE_K_PER_M = 0.0065
R_AIR = 287.05
G = 9.80665

# MoA reference elevations [m AMSL]
HAWASSA_AMSL_M = 1708.0
ADDIS_HIGHLAND_AMSL_M = 2300.0
HIGHLAND_DERATE_THRESHOLD_M = 2300.0

# Default heavy-lift spray platform (matches AircraftEnergyParams 30 kg MTOW class)
# ~1.35× hover weight at 30 kg — leaves measurable margin for derating at 2,300 m
DEFAULT_MAX_THRUST_SEA_LEVEL_N = 400.0
DEFAULT_A_MAX_NOMINAL_M_S2 = 5.0
DEFAULT_A_MAX_STRUCTURAL_CAP_M_S2 = 5.0
DEFAULT_AUTHORITY_BUFFER = 0.85


@dataclass(frozen=True)
class FlightEnvironment:
    """Operational site conditions for density and derating."""

    altitude_m_amsl: float = HAWASSA_AMSL_M
    temperature_c: float = 15.0
    max_thrust_sea_level_n: float = DEFAULT_MAX_THRUST_SEA_LEVEL_N
    a_max_nominal_m_s2: float = DEFAULT_A_MAX_NOMINAL_M_S2
    a_max_structural_cap_m_s2: float = DEFAULT_A_MAX_STRUCTURAL_CAP_M_S2
    authority_buffer: float = DEFAULT_AUTHORITY_BUFFER


def air_density_isa_kg_m3(
    altitude_m_amsl: float,
    temperature_c: float | None = None,
) -> float:
    """
    Air density [kg/m³] from ISA pressure altitude with optional ground temp offset.

    Uses troposphere lapse-rate pressure relation; temperature at altitude is
    T_sl + (temperature_c - 15) - L*h when temperature_c is set (15 °C = ISA sea level).
    """
    h = max(0.0, altitude_m_amsl)
    if h > 11_000.0:
        raise ValueError("ISA troposphere model valid only below 11,000 m AMSL")

    if temperature_c is None:
        t_k = T_0_K - LAPSE_RATE_K_PER_M * h
    else:
        t_sl_k = temperature_c + 273.15
        t_k = t_sl_k - LAPSE_RATE_K_PER_M * h

    if t_k <= 0:
        raise ValueError("Temperature at altitude must be positive")

    p_ratio = (1.0 - (LAPSE_RATE_K_PER_M * h) / T_0_K) ** (G / (LAPSE_RATE_K_PER_M * R_AIR))
    p_pa = P_0_PA * p_ratio
    return p_pa / (R_AIR * t_k)


def density_scaling_factors(rho: float) -> tuple[float, float]:
    """
    Return (profile_parasite_scale, induced_scale) relative to sea-level coeffs.

    P_induced ∝ 1/sqrt(rho)  → scale c2 by sqrt(rho_0 / rho)
    P_profile, P_parasite ∝ rho → scale c1, c3 by rho / rho_0
    """
    rho = max(rho, 0.5)
    parasite_scale = rho / RHO_0_KG_M3
    induced_scale = math.sqrt(RHO_0_KG_M3 / rho)
    return parasite_scale, induced_scale


def scale_energy_params_for_density(
    base: AircraftEnergyParams,
    rho: float,
) -> AircraftEnergyParams:
    """Return altitude-adjusted power coefficients for P(m, v)."""
    parasite_scale, induced_scale = density_scaling_factors(rho)
    return AircraftEnergyParams(
        c1=base.c1 * parasite_scale,
        c2=base.c2 * induced_scale,
        c3=base.c3 * parasite_scale,
        dry_mass_kg=base.dry_mass_kg,
        payload_mass_kg=base.payload_mass_kg,
        fluid_density_kg_per_l=base.fluid_density_kg_per_l,
        v_min=base.v_min,
        v_max=base.v_max,
    )


def derated_lateral_acceleration(
    total_mass_kg: float,
    env: FlightEnvironment | None = None,
) -> float:
    """
    Safe lateral acceleration cap [m/s²] from thrust margin at altitude.

    T_max(h) = T_max_ref * (rho / rho_0)
    a_max_safe = buffer * (T_max - mg) / m
    """
    e = env or FlightEnvironment()
    rho = air_density_isa_kg_m3(e.altitude_m_amsl, e.temperature_c)
    t_max = e.max_thrust_sea_level_n * (rho / RHO_0_KG_M3)
    weight = total_mass_kg * G

    if t_max <= weight:
        raise ValueError(
            "Vehicle is structurally stalled: zero excess thrust margin at "
            f"{e.altitude_m_amsl:.0f} m AMSL with {total_mass_kg:.1f} kg."
        )

    excess = t_max - weight
    a_safe = (excess / total_mass_kg) * e.authority_buffer
    return float(
        min(a_safe, e.a_max_nominal_m_s2, e.a_max_structural_cap_m_s2)
    )


def get_highland_constraints(
    total_mass_kg: float,
    env: FlightEnvironment | None = None,
    base_energy_params: AircraftEnergyParams | None = None,
) -> dict:
    """
    Density-aware planner constraints for arc_smoother / energy scheduling.

    Returns rho, a_max_lateral, scaled AircraftEnergyParams, hover_power_scaler.
    """
    e = env or FlightEnvironment()
    base = base_energy_params or AircraftEnergyParams()
    rho = air_density_isa_kg_m3(e.altitude_m_amsl, e.temperature_c)
    a_max = derated_lateral_acceleration(total_mass_kg, e)
    scaled_params = scale_energy_params_for_density(base, rho)
    _, induced_scale = density_scaling_factors(rho)

    return {
        "rho": rho,
        "a_max_lateral": a_max,
        "energy_params": scaled_params,
        "hover_power_scaler": induced_scale,
        "altitude_m_amsl": e.altitude_m_amsl,
        "highland_mode": e.altitude_m_amsl >= HIGHLAND_DERATE_THRESHOLD_M,
    }


def min_turn_speed_for_radius(
    radius_m: float,
    a_max_lateral: float,
    v_cap: float,
) -> float:
    """v = sqrt(a * r) — minimum speed at which turn fits lateral cap."""
    if radius_m <= 0.01:
        return 0.0
    return min(math.sqrt(a_max_lateral * radius_m), v_cap)
