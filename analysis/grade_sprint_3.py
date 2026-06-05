"""
Sprint 3 SITL grader — zero off-plot spraying (smallholder geofence).

Usage:
  PYTHONPATH=. python analysis/grade_sprint_3.py --latest /path/to/PX4-Autopilot
  PYTHONPATH=. python analysis/grade_sprint_3.py flight.ulg --meta analysis/output/sprint3_mission_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from pyulog import ULog
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from analysis.ulog_parser import get_latest_log


def _load_active_zone(meta: dict) -> MultiPolygon:
    plots = []
    for key in ('plot_a_utm', 'plot_b_utm'):
        if key not in meta:
            raise ValueError(f"Meta missing {key}")
        ring = meta[key]
        plots.append(Polygon(ring))
    merged = unary_union(plots)
    if merged.geom_type == 'Polygon':
        return MultiPolygon([merged])
    return merged


def _ned_to_utm(north: float, east: float, meta: dict) -> tuple[float, float]:
    """PX4 local NED (x=north, y=east) → mission UTM easting/northing."""
    e0 = float(meta['home_utm_e'])
    n0 = float(meta['home_utm_n'])
    return e0 + east, n0 + north


def _armed_mask(ulog: ULog, time_us: np.ndarray) -> np.ndarray:
    available = {d.name for d in ulog.data_list}
    if 'actuator_armed' not in available:
        return np.ones(len(time_us), dtype=bool)
    aa = ulog.get_dataset('actuator_armed').data
    aa_t = np.asarray(aa['timestamp'])
    aa_armed = np.asarray(aa['armed'], dtype=bool)
    idx = np.searchsorted(aa_t, time_us, side='left')
    idx = np.clip(idx, 0, len(aa_t) - 1)
    idx_l = np.clip(idx - 1, 0, len(aa_t) - 1)
    pick_l = np.abs(aa_t[idx_l] - time_us) < np.abs(aa_t[idx] - time_us)
    closest = np.where(pick_l, idx_l, idx)
    return aa_armed[closest]


def _in_corridor(easting: float, meta: dict) -> bool:
    corridor = meta.get('corridor_x_utm')
    if not corridor or len(corridor) != 2:
        return False
    return float(corridor[0]) <= easting <= float(corridor[1])


def _spray_intent_at(
    easting: float,
    northing: float,
    planned: list[dict],
    meta: dict,
    max_spray_dist_m: float = 15.0,
) -> bool:
    """
    Spray intent when closer to a spray waypoint than any transit waypoint.

    Corridor samples are always transit (pump OFF).
    """
    if _in_corridor(easting, meta):
        return False

    best_spray_d = float('inf')
    best_transit_d = float('inf')
    is_spray = False
    for wp in planned:
        d = np.hypot(easting - wp['x'], northing - wp['y'])
        if wp['is_spraying']:
            if d < best_spray_d:
                best_spray_d = d
                is_spray = True
        elif d < best_transit_d:
            best_transit_d = d

    if not is_spray or best_spray_d > max_spray_dist_m:
        return False
    return best_spray_d <= best_transit_d


def _mission_window_mask(
    north: np.ndarray,
    east: np.ndarray,
    meta: dict,
    zone: MultiPolygon,
    margin_m: float = 50.0,
) -> np.ndarray:
    """Ignore samples before the vehicle first enters the Sprint 3 envelope."""
    bounds = zone.bounds
    minx, miny, maxx, maxy = bounds
    in_envelope = np.zeros(len(north), dtype=bool)
    started = False
    for i in range(len(north)):
        utm_e, utm_n = _ned_to_utm(north[i], east[i], meta)
        inside = (
            (minx - margin_m <= utm_e <= maxx + margin_m)
            and (miny - margin_m <= utm_n <= maxy + margin_m)
        )
        if inside:
            started = True
        in_envelope[i] = started
    return in_envelope


def grade_planned_mission(meta: dict) -> dict:
    """Analytic check: all planned spray waypoints lie inside active zones."""
    zone = _load_active_zone(meta)
    tol = float(meta.get('overspray_tolerance_m', 1.0))
    allowed = zone.buffer(tol)
    planned = meta.get('planned_waypoints_utm', [])
    if not planned:
        raise ValueError("Meta missing planned_waypoints_utm")

    spray_count = 0
    off_plot = 0
    corridor_spray = 0
    for wp in planned:
        if not wp['is_spraying']:
            continue
        spray_count += 1
        pt = Point(wp['x'], wp['y'])
        if _in_corridor(wp['x'], meta):
            corridor_spray += 1
        if not allowed.contains(pt):
            off_plot += 1

    fraction = (off_plot / spray_count * 100.0) if spray_count else 0.0
    passed = spray_count > 0 and fraction == 0.0 and corridor_spray == 0
    return {
        'passed': passed,
        'mode': 'planned_only',
        'spray_samples': spray_count,
        'off_plot_samples': off_plot,
        'overspray_fraction_pct': fraction,
        'corridor_spray_samples': corridor_spray,
        'overspray_tolerance_m': tol,
    }


def grade_overspray(
    ulog_path: str,
    meta: dict,
    *,
    min_speed_m_s: float = 2.0,
) -> dict:
    ulog = ULog(ulog_path)
    zone = _load_active_zone(meta)
    tol = float(meta.get('overspray_tolerance_m', 1.0))
    allowed = zone.buffer(tol)

    planned = meta.get('planned_waypoints_utm', [])
    if not planned:
        raise ValueError("Meta missing planned_waypoints_utm — re-run run_sitl_mission.py")

    pos = ulog.get_dataset('vehicle_local_position').data
    t_us = np.asarray(pos['timestamp'])
    north = np.asarray(pos['x'], dtype=float)
    east = np.asarray(pos['y'], dtype=float)
    armed = _armed_mask(ulog, t_us)

    dt = np.diff(t_us) / 1e6
    dn = np.diff(north)
    de = np.diff(east)
    speed = np.hypot(dn, de) / np.maximum(dt, 1e-3)
    moving = np.concatenate([[False], armed[1:] & (speed > min_speed_m_s)])
    in_window = _mission_window_mask(north, east, meta, zone)

    spray_samples = 0
    off_plot = 0
    corridor_intrusion = 0

    active = moving & in_window
    for i in np.where(active)[0]:
        utm_e, utm_n = _ned_to_utm(north[i], east[i], meta)
        if _in_corridor(utm_e, meta):
            if _spray_intent_at(utm_e, utm_n, planned, meta):
                corridor_intrusion += 1
            continue
        if not _spray_intent_at(utm_e, utm_n, planned, meta):
            continue
        spray_samples += 1
        if not allowed.contains(Point(utm_e, utm_n)):
            off_plot += 1

    fraction = (off_plot / spray_samples * 100.0) if spray_samples else 0.0
    passed = spray_samples > 0 and fraction == 0.0 and corridor_intrusion == 0

    strict_tol = float(meta.get('kinematic_tolerance_m', 1.0))
    strict_allowed = zone.buffer(strict_tol)
    strict_off = 0
    for i in np.where(active)[0]:
        utm_e, utm_n = _ned_to_utm(north[i], east[i], meta)
        if not _spray_intent_at(utm_e, utm_n, planned, meta):
            continue
        if not strict_allowed.contains(Point(utm_e, utm_n)):
            strict_off += 1
    strict_fraction = (
        (strict_off / spray_samples * 100.0) if spray_samples else 0.0
    )

    return {
        'passed': passed,
        'spray_samples': int(spray_samples),
        'off_plot_samples': int(off_plot),
        'overspray_fraction_pct': fraction,
        'corridor_spray_samples': int(corridor_intrusion),
        'overspray_tolerance_m': tol,
        'strict_1m_overspray_fraction_pct': strict_fraction,
        'drift_envelope_m': meta.get('drift_envelope_m'),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Grade Sprint 3 geofence (zero overspray).')
    parser.add_argument('ulog', nargs='?', help='Path to .ulg file')
    parser.add_argument('--latest', metavar='PX4_ROOT', help='Newest log under PX4 build tree')
    parser.add_argument(
        '--meta',
        default='analysis/output/sprint3_mission_meta.json',
        help='Sprint 3 sidecar from run_sitl_mission.py',
    )
    parser.add_argument('--json-out', default='analysis/output/sprint3_grade.json')
    parser.add_argument(
        '--planned-only',
        action='store_true',
        help='Grade planned waypoints in meta only (no ULog)',
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.meta):
        print(f"Missing meta: {args.meta}", file=sys.stderr)
        return 2

    with open(args.meta, encoding='utf-8') as f:
        meta = json.load(f)

    if args.planned_only:
        result = grade_planned_mission(meta)
        ulg_path = None
    elif args.latest:
        log_dir = os.path.join(args.latest, 'build', 'px4_sitl_default', 'rootfs', 'log')
        ulg_path = get_latest_log(log_dir)
    elif args.ulog:
        ulg_path = args.ulog
    else:
        parser.error('Provide ulog path, --latest PX4_ROOT, or --planned-only')

    if not args.planned_only:
        result = grade_overspray(ulg_path, meta)
    os.makedirs(os.path.dirname(args.json_out) or '.', exist_ok=True)
    payload = {'ulog_path': ulg_path, **result}
    with open(args.json_out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    title = ulg_path if ulg_path else f"planned mission ({args.meta})"
    print(f"\nSprint 3 Geofence Grade: {title}")
    print("-" * 60)
    icon = "✅ PASS" if result['passed'] else "❌ FAIL"
    print(
        f"{icon}  Overspray fraction: {result['overspray_fraction_pct']:.1f}% "
        f"({result['off_plot_samples']}/{result['spray_samples']} off-plot samples) "
        f"@ {result.get('overspray_tolerance_m', 1.0):.1f} m tolerance"
    )
    if 'strict_1m_overspray_fraction_pct' in result:
        print(
            f"      Strict 1.0 m margin: {result['strict_1m_overspray_fraction_pct']:.1f}% "
            f"(Sprint 1 drift command envelope; see Volume II §5.7)"
        )
    print(f"      Corridor spray samples: {result['corridor_spray_samples']} (must be 0)")
    print("-" * 60)
    if result['passed']:
        print("✅ SPRINT 3 GEOFENCE VALIDATION PASSED (0.0% overspray)")
    else:
        print("❌ SPRINT 3 GEOFENCE VALIDATION FAILED")
    print(f"Wrote grade report → {args.json_out}")
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
