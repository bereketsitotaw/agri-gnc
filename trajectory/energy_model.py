"""
Mass-varying rotary-wing energy model for agricultural spray missions.

Minimizes cost of transport E_meter = P(m, v) / v where:

    P(m, v) = c1 * m^1.5 + c2 * m^2 / v + c3 * v^3

  c1: profile power coefficient (rotor disk loading)
  c2: induced power coefficient (lift / momentum theory)
  c3: parasitic power coefficient (airframe drag)
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AircraftEnergyParams:
    """Quadrotor-specific aerodynamic constants and mass bounds."""

    # Tuned for v* ~ 10 m/s @ 30 kg and ~ 7 m/s @ 15 kg within [5, 12] m/s
    c1: float = 2.0
    c2: float = 20.0
    c3: float = 2.0
    dry_mass_kg: float = 15.0
    payload_mass_kg: float = 15.0
    fluid_density_kg_per_l: float = 1.0
    v_min: float = 3.0
    v_max: float = 12.0


def total_mass_kg(
    tank_fraction: float,
    params: AircraftEnergyParams | None = None,
) -> float:
    """Remaining total mass from tank fill level in [0, 1]."""
    p = params or AircraftEnergyParams()
    fraction = float(np.clip(tank_fraction, 0.0, 1.0))
    return p.dry_mass_kg + fraction * p.payload_mass_kg


def instantaneous_power_w(
    mass_kg: float,
    speed_m_s: float,
    params: AircraftEnergyParams | None = None,
) -> float:
    """
    P(m, v) = c1*m^1.5 + c2*m^2/v + c3*v^3  [Watts]
    """
    p = params or AircraftEnergyParams()
    v = max(speed_m_s, 0.5)
    m = max(mass_kg, p.dry_mass_kg)
    return (
        p.c1 * (m ** 1.5)
        + p.c2 * (m ** 2) / v
        + p.c3 * (v ** 3)
    )


def energy_per_meter(
    mass_kg: float,
    speed_m_s: float,
    params: AircraftEnergyParams | None = None,
) -> float:
    """Cost of transport: J/m (energy to fly one meter at speed v)."""
    v = max(speed_m_s, 0.5)
    return instantaneous_power_w(mass_kg, v, params) / v


def optimal_cruise_speed(
    mass_kg: float,
    params: AircraftEnergyParams | None = None,
    num_samples: int = 200,
) -> float:
    """
    v* = argmin_v  P(m, v) / v  over [v_min, v_max].

    Heavy aircraft: optimal speed tends higher (induced term dominates).
    Light aircraft: optimal speed tends lower (parasitic term dominates).
    """
    p = params or AircraftEnergyParams()
    speeds = np.linspace(p.v_min, p.v_max, num_samples)
    costs = [energy_per_meter(mass_kg, v, p) for v in speeds]
    return float(speeds[int(np.argmin(costs))])


def spray_volume_liters(
    sweep_length_m: float,
    swath_width_m: float,
    application_rate_l_per_m2: float,
) -> float:
    """Liters applied along one spray pass."""
    area_m2 = sweep_length_m * swath_width_m
    return area_m2 * application_rate_l_per_m2


def mass_after_spray_pass(
    current_mass_kg: float,
    sweep_length_m: float,
    swath_width_m: float,
    application_rate_l_per_m2: float,
    params: AircraftEnergyParams | None = None,
) -> float:
    """Reduce mass after one spraying sweep by fluid consumed."""
    p = params or AircraftEnergyParams()
    volume_l = spray_volume_liters(sweep_length_m, swath_width_m, application_rate_l_per_m2)
    mass_lost = volume_l * p.fluid_density_kg_per_l
    min_mass = p.dry_mass_kg
    return max(min_mass, current_mass_kg - mass_lost)


def sweep_mass_schedule(
    segments: list,
    swath_width_m: float,
    application_rate_l_per_m2: float = 0.02,
    params: AircraftEnergyParams | None = None,
) -> list[float]:
    """
    Estimated total mass [kg] at the start of each spray segment.

    segments: list of LineString sweep segments (in order).
    """
    p = params or AircraftEnergyParams()
    mass = p.dry_mass_kg + p.payload_mass_kg
    schedule = []

    for seg in segments:
        schedule.append(mass)
        coords = list(seg.coords)
        if len(coords) < 2:
            continue
        length = 0.0
        for i in range(len(coords) - 1):
            dx = coords[i + 1][0] - coords[i][0]
            dy = coords[i + 1][1] - coords[i][1]
            length += math.hypot(dx, dy)
        mass = mass_after_spray_pass(
            mass, length, swath_width_m, application_rate_l_per_m2, p
        )

    return schedule


def mission_energy_joules(
    segments: list,
    swath_width_m: float,
    application_rate_l_per_m2: float = 0.02,
    params: AircraftEnergyParams | None = None,
) -> float:
    """
    Approximate total mission energy [J] using per-sweep optimal cruise speeds.
    """
    p = params or AircraftEnergyParams()
    masses = sweep_mass_schedule(segments, swath_width_m, application_rate_l_per_m2, p)
    total = 0.0

    for seg, mass in zip(segments, masses):
        coords = list(seg.coords)
        if len(coords) < 2:
            continue
        length = 0.0
        for i in range(len(coords) - 1):
            dx = coords[i + 1][0] - coords[i][0]
            dy = coords[i + 1][1] - coords[i][1]
            length += math.hypot(dx, dy)
        v_opt = optimal_cruise_speed(mass, p)
        total += energy_per_meter(mass, v_opt, p) * length

    return total
