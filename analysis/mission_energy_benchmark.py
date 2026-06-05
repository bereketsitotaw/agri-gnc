"""
Analytical mission energy benchmark: optimal v* vs constant cruise.

Compares integrated spray-leg energy from energy_model.py on the same CPP
geometry. Use after planning any field to populate thesis energy tables.

Example:
  PYTHONPATH=. python3 analysis/mission_energy_benchmark.py
  PYTHONPATH=. python3 analysis/mission_energy_benchmark.py --constant-speed 8.0
"""

from __future__ import annotations

import argparse
import math
import sys

from shapely.geometry import Polygon

from cpp.sweep_optimizer import optimize_sweep_angle
from geometry.polygon_utils import validate_and_clean_polygon, project_to_utm
from trajectory.energy_model import (
    AircraftEnergyParams,
    energy_per_meter,
    mission_energy_joules,
    optimal_cruise_speed,
    sweep_mass_schedule,
)

# Hawassa SITL test field (same as run_sitl_mission.py)
DEFAULT_WGS84 = [
    (38.470000, 7.050000),
    (38.471000, 7.050000),
    (38.471000, 7.051000),
    (38.470000, 7.051000),
]


def segment_length_m(segment) -> float:
    coords = list(segment.coords)
    return sum(
        math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
        for i in range(len(coords) - 1)
    )


def mission_energy_constant_joules(
    segments: list,
    cruise_speed: float,
    swath_width_m: float,
    application_rate_l_per_m2: float,
    params: AircraftEnergyParams,
) -> tuple[float, float, list[dict]]:
    """Spray-leg energy and time at fixed cruise speed."""
    masses = sweep_mass_schedule(segments, swath_width_m, application_rate_l_per_m2, params)
    total_e = total_t = 0.0
    rows: list[dict] = []

    for idx, (seg, mass) in enumerate(zip(segments, masses), start=1):
        length_m = segment_length_m(seg)
        v = cruise_speed
        e_m = energy_per_meter(mass, v, params)
        energy_j = e_m * length_m
        time_s = length_m / v
        total_e += energy_j
        total_t += time_s
        rows.append({
            'sweep': idx,
            'mass_kg': mass,
            'speed_m_s': v,
            'length_m': length_m,
            'energy_j': energy_j,
            'time_s': time_s,
        })

    return total_e, total_t, rows


def mission_energy_optimal_joules(
    segments: list,
    swath_width_m: float,
    application_rate_l_per_m2: float,
    params: AircraftEnergyParams,
) -> tuple[float, float, list[dict]]:
    """Spray-leg energy and time at per-sweep v*."""
    masses = sweep_mass_schedule(segments, swath_width_m, application_rate_l_per_m2, params)
    total_e = mission_energy_joules(segments, swath_width_m, application_rate_l_per_m2, params)
    total_t = 0.0
    rows: list[dict] = []

    for idx, (seg, mass) in enumerate(zip(segments, masses), start=1):
        length_m = segment_length_m(seg)
        v = optimal_cruise_speed(mass, params)
        e_m = energy_per_meter(mass, v, params)
        energy_j = e_m * length_m
        time_s = length_m / v
        total_t += time_s
        rows.append({
            'sweep': idx,
            'mass_kg': mass,
            'speed_m_s': v,
            'length_m': length_m,
            'energy_j': energy_j,
            'time_s': time_s,
        })

    return total_e, total_t, rows


def run_benchmark(
    wgs84_coords: list[tuple[float, float]],
    swath_width_m: float,
    application_rate_l_per_m2: float,
    constant_speed: float,
    angle_step: float,
    params: AircraftEnergyParams | None = None,
) -> dict:
    p = params or AircraftEnergyParams()
    poly_utm, _ = project_to_utm(validate_and_clean_polygon(Polygon(wgs84_coords)))
    _, segments, metrics = optimize_sweep_angle(poly_utm, swath_width_m, angle_step)

    e_opt, t_opt, rows_opt = mission_energy_optimal_joules(
        segments, swath_width_m, application_rate_l_per_m2, p
    )
    e_const, t_const, rows_const = mission_energy_constant_joules(
        segments, constant_speed, swath_width_m, application_rate_l_per_m2, p
    )

    delta_e = e_const - e_opt
    return {
        'path_length_m': metrics['total_length'],
        'turn_count': metrics['turn_count'],
        'n_segments': len(segments),
        'e_opt_j': e_opt,
        'e_const_j': e_const,
        'delta_e_j': delta_e,
        'pct_saved': 100.0 * delta_e / e_const if e_const else 0.0,
        't_opt_s': t_opt,
        't_const_s': t_const,
        'delta_t_s': t_const - t_opt,
        'rows_opt': rows_opt,
        'rows_const': rows_const,
        'constant_speed': constant_speed,
    }


def print_markdown_table(result: dict) -> None:
    r = result
    print("\n## Mission Energy Benchmark\n")
    print(f"| Parameter | Value |")
    print(f"|-----------|-------|")
    print(f"| Path length | {r['path_length_m']:.1f} m |")
    print(f"| Spray segments | {r['n_segments']} |")
    print(f"| Turns (CPP) | {r['turn_count']} |")
    print(f"| Constant baseline | {r['constant_speed']:.1f} m/s |")
    print()
    print("| Policy | Spray energy | Spray-leg time |")
    print("|--------|--------------|----------------|")
    print(f"| Energy-optimal v* | {r['e_opt_j']/1000:.2f} kJ ({r['e_opt_j']:.0f} J) | {r['t_opt_s']:.1f} s |")
    print(f"| Constant cruise | {r['e_const_j']/1000:.2f} kJ ({r['e_const_j']:.0f} J) | {r['t_const_s']:.1f} s |")
    print()
    print(f"**ΔE (constant − optimal):** {r['delta_e_j']/1000:.2f} kJ ({r['delta_e_j']:.0f} J) — **{r['pct_saved']:.1f}% saved**")
    sign = "faster" if r['delta_t_s'] > 0 else "slower"
    print(f"**Spray-leg time:** optimal is {abs(r['delta_t_s']):.1f} s {sign} than constant\n")

    print("### Per-sweep detail\n")
    print("| Sweep | Mass (kg) | v* (m/s) | v_const (m/s) | E_opt (kJ) | E_const (kJ) |")
    print("|-------|-----------|----------|---------------|------------|--------------|")
    for ro, rc in zip(r['rows_opt'], r['rows_const']):
        print(
            f"| {ro['sweep']} | {ro['mass_kg']:.1f} | {ro['speed_m_s']:.1f} | "
            f"{rc['speed_m_s']:.1f} | {ro['energy_j']/1000:.2f} | {rc['energy_j']/1000:.2f} |"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analytical mission energy benchmark")
    parser.add_argument("--swath", type=float, default=30.0, help="Swath width [m]")
    parser.add_argument("--app-rate", type=float, default=0.02, help="Application rate [L/m²]")
    parser.add_argument("--constant-speed", type=float, default=8.0, help="Baseline cruise [m/s]")
    parser.add_argument("--angle-step", type=float, default=5.0, help="CPP angle search step [deg]")
    args = parser.parse_args(argv)

    result = run_benchmark(
        DEFAULT_WGS84,
        swath_width_m=args.swath,
        application_rate_l_per_m2=args.app_rate,
        constant_speed=args.constant_speed,
        angle_step=args.angle_step,
    )
    print_markdown_table(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
