"""
Phase I SITL validation — automated ULog grading (Sprints 1 & 2).

Usage:
  PYTHONPATH=. python analysis/grade_phase_1.py /path/to/log.ulg
  PYTHONPATH=. python analysis/grade_phase_1.py --latest /path/to/PX4-Autopilot

Drift test (Sprint 1):
  --wind-speed 5 --wind-from 180 --cruise-speed 8 --path-heading-deg 0

Highland test (Sprint 2):
  --pwm-max 1950 --thrust-ceiling-pct 85
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field

import numpy as np
from pyulog import ULog

from analysis.energy_parser import parse_battery_energy
from analysis.ulog_parser import compute_xte, get_latest_log, parse_local_position, parse_position_setpoint
from trajectory.altitude_physics import (
    ADDIS_HIGHLAND_AMSL_M,
    DEFAULT_A_MAX_NOMINAL_M_S2,
    FlightEnvironment,
    air_density_isa_kg_m3,
    derated_lateral_acceleration,
)
from trajectory.drift_model import crab_angle_rad, vehicle_yaw_deg_from_track
from trajectory.energy_model import AircraftEnergyParams


# PX4 PWM microseconds (typical ESC range)
PWM_ABSOLUTE_MAX = 2000
PWM_DEFAULT_FAIL = 1950
# PX4 actuator_motors.control[i] is normalized [0, 1]
MOTOR_CONTROL_FAIL = 0.85


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class Phase1GradeReport:
    ulog_path: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        required = [c for c in self.checks if c.name != 'battery_info']
        return all(c.passed for c in required)


def quaternion_to_yaw_rad(q: np.ndarray) -> np.ndarray:
    """Yaw from body quaternion [w, x, y, z] in NED (PX4 vehicle_attitude)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _dataset_or_none(ulog: ULog, name: str):
    available = {d.name for d in ulog.data_list}
    return ulog.get_dataset(name) if name in available else None


def _channel_arrays_from_fields(data: dict, prefix: str) -> list[np.ndarray]:
    """Collect PX4 multi-channel fields like output[0], control[3]."""
    arrays = []
    for key, val in data.items():
        if key.startswith(prefix) and '[' in key:
            arr = np.asarray(val, dtype=float)
            if arr.size > 0:
                arrays.append(arr)
    return arrays


def parse_actuator_pwm(ulog: ULog) -> dict:
    """
    Extract motor output statistics from PX4 ULog topics.

    Supports:
    - actuator_outputs.output[i] — SITL often 0–~900 (not 2000 µs)
    - actuator_motors.control[i] — normalized 0–1
    """
    for topic in ('actuator_outputs', 'actuator_servos', 'actuator_motors'):
        ds = _dataset_or_none(ulog, topic)
        if ds is None:
            continue
        data = ds.data

        if topic == 'actuator_motors':
            arrays = _channel_arrays_from_fields(data, 'control[')
            if not arrays:
                continue
            stacked = np.column_stack(arrays)
            flat = stacked.reshape(-1)
            valid = flat[np.isfinite(flat)]
            if valid.size == 0:
                continue
            return {
                'topic': topic,
                'scale': 'normalized_0_1',
                'max_value': float(np.max(valid)),
                'p95_value': float(np.percentile(valid, 95)),
                'mean_value': float(np.mean(valid)),
                'channels': len(arrays),
            }

        arrays = _channel_arrays_from_fields(data, 'output[')
        if not arrays and 'output' in data:
            outputs = np.asarray(data['output'], dtype=float)
            if outputs.ndim == 1:
                arrays = [outputs]
            else:
                arrays = [outputs[:, i] for i in range(outputs.shape[1])]

        if not arrays:
            continue

        stacked = np.column_stack(arrays)
        flat = stacked.reshape(-1)
        valid = flat[np.isfinite(flat) & (flat > 1e-6)]
        if valid.size == 0:
            continue

        max_val = float(np.max(valid))
        scale = 'pwm_us' if max_val > 100 else 'normalized_0_1'
        return {
            'topic': topic,
            'scale': scale,
            'max_value': max_val,
            'p95_value': float(np.percentile(valid, 95)),
            'mean_value': float(np.mean(valid)),
            'channels': len(arrays),
        }

    return {'topic': None, 'scale': None, 'max_value': float('nan')}


def _armed_interval(ulog: ULog) -> tuple[np.ndarray, np.ndarray] | None:
    ds = _dataset_or_none(ulog, 'actuator_armed')
    if ds is None:
        return None
    t = np.asarray(ds.data['timestamp'])
    armed = np.asarray(ds.data['armed'], dtype=bool)
    return t, armed


def _samples_while_armed(ulog: ULog, sample_times_us: np.ndarray) -> np.ndarray:
    interval = _armed_interval(ulog)
    if interval is None:
        return np.ones(len(sample_times_us), dtype=bool)
    aa_t, aa_armed = interval
    idx = np.searchsorted(aa_t, sample_times_us, side='left')
    idx = np.clip(idx, 0, len(aa_t) - 1)
    idx_l = np.clip(idx - 1, 0, len(aa_t) - 1)
    pick_l = np.abs(aa_t[idx_l] - sample_times_us) < np.abs(aa_t[idx] - sample_times_us)
    closest = np.where(pick_l, idx_l, idx)
    return aa_armed[closest]


def estimate_dominant_path_heading_rad(ulog: ULog) -> float | None:
    """Infer mean spray-leg heading from horizontal velocity (NED x=North, y=East)."""
    ds = _dataset_or_none(ulog, 'vehicle_local_position')
    if ds is None:
        return None
    d = ds.data
    t = np.asarray(d['timestamp'])
    x = np.asarray(d['x'], dtype=float)
    y = np.asarray(d['y'], dtype=float)
    armed = _samples_while_armed(ulog, t)

    dt = np.diff(t) / 1e6
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.hypot(dx, dy) / np.maximum(dt, 1e-3)
    mask = armed[1:] & (speed > 2.0)
    if not np.any(mask):
        return None
    headings = np.arctan2(dy[mask], dx[mask])
    return float(np.median(headings))


def parse_attitude_yaw(ulog: ULog) -> tuple[np.ndarray, np.ndarray]:
    ds = _dataset_or_none(ulog, 'vehicle_attitude')
    if ds is None:
        raise ValueError("ULog missing vehicle_attitude")

    d = ds.data
    t = np.asarray(d['timestamp'])
    q = np.vstack([
        np.asarray(d['q[0]']),
        np.asarray(d['q[1]']),
        np.asarray(d['q[2]']),
        np.asarray(d['q[3]']),
    ])
    yaw = quaternion_to_yaw_rad(q)
    time_s = (t - t[0]) / 1e6
    return time_s, yaw


def grade_pwm_saturation(
    ulog: ULog,
    pwm_fail_threshold: float = PWM_DEFAULT_FAIL,
    motor_control_fail: float = MOTOR_CONTROL_FAIL,
    thrust_ceiling_pct: float = 85.0,
) -> CheckResult:
    """Sprint 2: motors must stay below saturation during armed flight."""
    act = parse_actuator_pwm(ulog)
    topic = act['topic']
    max_val = act.get('max_value', float('nan'))

    if topic is None or not np.isfinite(max_val):
        return CheckResult(
            name='pwm_saturation',
            passed=False,
            message='No actuator output topic found (actuator_outputs / actuator_motors).',
            details=act,
        )

    scale = act.get('scale', 'pwm_us')
    if scale == 'normalized_0_1':
        fail_at = motor_control_fail
        passed = max_val < fail_at
        msg = (
            f"Max motor control {max_val:.3f} on '{topic}' "
            + ('<' if passed else '>=')
            + f" fail threshold {fail_at:.2f} ({thrust_ceiling_pct:.0f}% ceiling)."
        )
    else:
        fail_at = pwm_fail_threshold
        passed = max_val < fail_at
        ceiling_pwm = PWM_ABSOLUTE_MAX * (thrust_ceiling_pct / 100.0)
        msg = (
            f"Max actuator output {max_val:.0f} on '{topic}' ({scale}) "
            + ('<' if passed else '>=')
            + f" fail threshold {fail_at:.0f} "
            f"(~{ceiling_pwm:.0f} at {thrust_ceiling_pct:.0f}% of {PWM_ABSOLUTE_MAX})."
        )

    return CheckResult(
        name='pwm_saturation',
        passed=passed,
        message=msg,
        details={**act, 'fail_threshold': fail_at},
    )


def grade_xte_tracking(
    ulog: ULog,
    max_xte_m: float = 0.97,
) -> CheckResult:
    """XTE vs commanded setpoint (offset path is what PX4 tracks)."""
    actual = parse_local_position(ulog)
    available = {d.name for d in ulog.data_list}
    if 'vehicle_local_position_setpoint' not in available:
        return CheckResult(
            name='xte_tracking',
            passed=False,
            message='Missing vehicle_local_position_setpoint — cannot compute XTE.',
        )

    setpoint = parse_position_setpoint(ulog)
    xte_df = compute_xte(actual, setpoint)
    max_xte = float(xte_df['xte_m'].max())
    rmse = float(np.sqrt(np.mean(xte_df['xte_m'] ** 2)))
    passed = max_xte <= max_xte_m

    return CheckResult(
        name='xte_tracking',
        passed=passed,
        message=f"Max XTE {max_xte:.3f} m (limit {max_xte_m:.2f} m), RMSE {rmse:.3f} m.",
        details={'max_xte_m': max_xte, 'rmse_m': rmse, 'samples': len(xte_df)},
    )


def grade_crab_angle(
    ulog: ULog,
    wind_speed: float,
    wind_from_deg: float,
    cruise_speed_m_s: float,
    path_heading_rad: float | None = None,
    yaw_tolerance_deg: float = 12.0,
    min_armed_samples: int = 50,
) -> CheckResult:
    """
    Sprint 1: compare median armed yaw to expected crab during horizontal flight.
    """
    if wind_speed <= 0:
        return CheckResult(
            name='crab_angle',
            passed=True,
            message='Wind speed zero — crab check skipped (calm air).',
        )

    heading = path_heading_rad
    if heading is None:
        heading = estimate_dominant_path_heading_rad(ulog)
    if heading is None:
        return CheckResult(
            name='crab_angle',
            passed=False,
            message='Could not infer path heading; pass --path-heading-deg.',
        )

    try:
        expected_crab_rad = crab_angle_rad(
            heading, wind_speed, wind_from_deg, cruise_speed_m_s
        )
    except ValueError as exc:
        return CheckResult(
            name='crab_angle',
            passed=False,
            message=str(exc),
        )

    expected_yaw_deg = vehicle_yaw_deg_from_track(heading, expected_crab_rad)

    att_ds = _dataset_or_none(ulog, 'vehicle_attitude')
    if att_ds is None:
        return CheckResult(
            name='crab_angle',
            passed=False,
            message='Missing vehicle_attitude.',
        )

    d = att_ds.data
    t_us = np.asarray(d['timestamp'])
    armed = _samples_while_armed(ulog, t_us)

    pos_ds = _dataset_or_none(ulog, 'vehicle_local_position')
    speed_mask = np.ones(len(t_us), dtype=bool)
    if pos_ds is not None:
        pd = pos_ds.data
        px = np.asarray(pd['x'], dtype=float)
        py = np.asarray(pd['y'], dtype=float)
        pt = np.asarray(pd['timestamp'])
        dt = np.diff(pt) / 1e6
        dx = np.diff(px)
        dy = np.diff(py)
        spd = np.hypot(dx, dy) / np.maximum(dt, 1e-3)
        spd_full = np.zeros(len(t_us))
        idx = np.searchsorted(pt, t_us, side='left')
        idx = np.clip(idx, 1, len(spd) - 1)
        spd_full = spd[idx - 1]
        speed_mask = spd_full > 2.0

    q = np.vstack([
        np.asarray(d['q[0]']),
        np.asarray(d['q[1]']),
        np.asarray(d['q[2]']),
        np.asarray(d['q[3]']),
    ])
    yaw_rad = quaternion_to_yaw_rad(q)
    yaw_deg = np.degrees(yaw_rad) % 360.0

    mask = armed & speed_mask
    if np.sum(mask) < min_armed_samples:
        mask = armed
    if np.sum(mask) < min_armed_samples:
        return CheckResult(
            name='crab_angle',
            passed=False,
            message=f'Insufficient attitude samples ({np.sum(mask)}).',
        )

    measured_yaw_deg = float(np.median(yaw_deg[mask]))

    def _angle_diff_deg(a: float, b: float) -> float:
        d = (a - b + 180.0) % 360.0 - 180.0
        return abs(d)

    err_deg = _angle_diff_deg(measured_yaw_deg, expected_yaw_deg)
    passed = err_deg <= yaw_tolerance_deg

    return CheckResult(
        name='crab_angle',
        passed=passed,
        message=(
            f"Yaw median {measured_yaw_deg:.1f}° vs expected {expected_yaw_deg:.1f}° "
            f"(err {err_deg:.1f}°, tol {yaw_tolerance_deg:.1f}°)."
        ),
        details={
            'measured_yaw_deg': measured_yaw_deg,
            'expected_yaw_deg': expected_yaw_deg,
            'expected_crab_rad': expected_crab_rad,
            'path_heading_rad': heading,
            'yaw_error_deg': err_deg,
        },
    )


def grade_highland_planner_bounds(
    meta: dict,
    *,
    expect_home_m: float | None = None,
) -> CheckResult:
    """
    Sprint 2: mission sidecar must reflect thin-air derating (rho, a_max, highland flag).
    """
    home = float(meta.get('home_amsl_m', 0.0))
    if expect_home_m is not None and abs(home - expect_home_m) > 1.0:
        return CheckResult(
            name='highland_planner',
            passed=False,
            message=(
                f"Mission meta home_amsl_m={home:.0f} m does not match "
                f"expected {expect_home_m:.0f} m — regenerate with HOME_AMSL_M."
            ),
            details={'home_amsl_m': home, 'expected_home_m': expect_home_m},
        )

    if home < ADDIS_HIGHLAND_AMSL_M - 1.0:
        return CheckResult(
            name='highland_planner',
            passed=False,
            message=(
                f"Highland test requires home_amsl_m >= {ADDIS_HIGHLAND_AMSL_M:.0f} m; "
                f"got {home:.0f} m."
            ),
        )

    if not meta.get('enable_altitude_derating', False):
        return CheckResult(
            name='highland_planner',
            passed=False,
            message='Altitude derating disabled in mission meta.',
        )

    rho_meta = meta.get('rho_kg_m3')
    a_meta = meta.get('a_max_lateral_m_s2')
    env = FlightEnvironment(altitude_m_amsl=home)
    params = AircraftEnergyParams()
    mass = params.dry_mass_kg + params.payload_mass_kg
    rho_calc = air_density_isa_kg_m3(home)
    a_calc = derated_lateral_acceleration(mass, env)

    rho = float(rho_meta) if rho_meta is not None else rho_calc
    a_max = float(a_meta) if a_meta is not None else a_calc
    highland = bool(meta.get('highland_mode', home >= ADDIS_HIGHLAND_AMSL_M))

    rho_ok = 0.92 <= rho <= 1.02
    a_ok = a_max < DEFAULT_A_MAX_NOMINAL_M_S2 and a_max <= a_calc * 1.05
    passed = highland and rho_ok and a_ok

    return CheckResult(
        name='highland_planner',
        passed=passed,
        message=(
            f"Highland planner: rho={rho:.3f} kg/m³, a_max={a_max:.2f} m/s² "
            f"(nominal cap {DEFAULT_A_MAX_NOMINAL_M_S2:.1f}), highland_mode={highland}."
        ),
        details={
            'home_amsl_m': home,
            'rho_kg_m3': rho,
            'a_max_lateral_m_s2': a_max,
            'a_max_calculated_m_s2': a_calc,
            'highland_mode': highland,
        },
    )


def grade_battery_voltage_margin(
    ulog: ULog,
    max_drop_v: float = 2.0,
) -> CheckResult:
    """Sprint 2: armed-flight voltage sag during high-demand turns (informational in SITL)."""
    ds = _dataset_or_none(ulog, 'battery_status')
    if ds is None:
        return CheckResult(
            name='battery_voltage',
            passed=True,
            message='No battery_status topic — voltage margin check skipped.',
        )

    d = ds.data
    t_us = np.asarray(d['timestamp'])
    voltage = np.asarray(d['voltage_v'], dtype=float)
    armed = _samples_while_armed(ulog, t_us)
    v = voltage[armed & np.isfinite(voltage)]
    if v.size < 10:
        v = voltage[np.isfinite(voltage)]
    if v.size < 2:
        return CheckResult(
            name='battery_voltage',
            passed=False,
            message='Insufficient battery voltage samples.',
        )

    v_med = float(np.median(v))
    v_min = float(np.min(v))
    drop = v_med - v_min
    passed = drop <= max_drop_v
    return CheckResult(
        name='battery_voltage',
        passed=passed,
        message=(
            f"Armed voltage median {v_med:.2f} V, min {v_min:.2f} V "
            f"(sag {drop:.2f} V, limit {max_drop_v:.1f} V)."
        ),
        details={
            'voltage_median_v': v_med,
            'voltage_min_v': v_min,
            'voltage_sag_v': drop,
            'max_drop_v': max_drop_v,
        },
    )


def grade_battery_note(ulog: ULog) -> CheckResult:
    """Informational — SITL battery is often a stub."""
    try:
        info = parse_battery_energy(ulog, armed_only=True)
        details = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in info.items()
            if k not in ('time_s', 'power_w')
        }
        return CheckResult(
            name='battery_info',
            passed=True,
            message=(
                f"SITL energy {info['energy_j']:.1f} J over {info['duration_s']:.1f} s "
                f"(treat as informational only)."
            ),
            details=details,
        )
    except Exception as exc:
        return CheckResult(
            name='battery_info',
            passed=True,
            message=f"Battery parse skipped: {exc}",
        )


def grade_phase_1_sitl(
    ulg_path: str,
    *,
    wind_speed: float = 5.0,
    wind_from_deg: float = 180.0,
    cruise_speed_m_s: float = 8.0,
    path_heading_deg: float | None = None,
    max_xte_m: float = 0.97,
    pwm_fail_threshold: float = PWM_DEFAULT_FAIL,
    motor_control_fail: float = MOTOR_CONTROL_FAIL,
    yaw_tolerance_deg: float = 12.0,
    run_drift: bool = True,
    run_saturation: bool = True,
    mission_meta: dict | None = None,
    expect_home_m: float | None = None,
) -> Phase1GradeReport:
    ulog = ULog(ulg_path)
    report = Phase1GradeReport(ulog_path=ulg_path)

    heading_rad = math.radians(path_heading_deg) if path_heading_deg is not None else None

    if run_drift:
        report.checks.append(grade_xte_tracking(ulog, max_xte_m=max_xte_m))
        report.checks.append(grade_crab_angle(
            ulog,
            wind_speed=wind_speed,
            wind_from_deg=wind_from_deg,
            cruise_speed_m_s=cruise_speed_m_s,
            path_heading_rad=heading_rad,
            yaw_tolerance_deg=yaw_tolerance_deg,
        ))

    if run_saturation:
        if mission_meta is not None:
            report.checks.append(grade_highland_planner_bounds(
                mission_meta,
                expect_home_m=expect_home_m,
            ))
        report.checks.append(grade_pwm_saturation(
            ulog,
            pwm_fail_threshold=pwm_fail_threshold,
            motor_control_fail=motor_control_fail,
        ))
        report.checks.append(grade_battery_voltage_margin(ulog))

    report.checks.append(grade_battery_note(ulog))
    return report


def print_report(report: Phase1GradeReport) -> None:
    print(f"\nPhase I SITL Grade: {report.ulog_path}")
    print("-" * 60)
    for check in report.checks:
        icon = "✅ PASS" if check.passed else "❌ FAIL"
        print(f"{icon}  [{check.name}] {check.message}")
    print("-" * 60)
    overall = "✅ PHASE I SITL VALIDATION PASSED" if report.passed else "❌ PHASE I SITL VALIDATION FAILED"
    print(overall + "\n")


def _json_sanitize(obj):
    """Convert numpy scalars/arrays for JSON export."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def save_report(report: Phase1GradeReport, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    payload = _json_sanitize({
        'ulog_path': report.ulog_path,
        'passed': report.passed,
        'checks': [asdict(c) for c in report.checks],
    })
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote grade report → {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Grade Phase I SITL flights from ULog.')
    parser.add_argument('ulog', nargs='?', help='Path to .ulg file')
    parser.add_argument('--latest', metavar='PX4_ROOT', help='Use newest log under PX4 build tree')
    parser.add_argument('--wind-speed', type=float, default=5.0)
    parser.add_argument('--wind-from', type=float, default=180.0, dest='wind_from')
    parser.add_argument('--cruise-speed', type=float, default=8.0)
    parser.add_argument('--path-heading-deg', type=float, default=None,
                        help='Dominant spray-leg heading (math: 0=East, 90=North)')
    parser.add_argument('--max-xte', type=float, default=0.97)
    parser.add_argument('--pwm-max', type=float, default=PWM_DEFAULT_FAIL,
                        help='Fail if actuator_outputs (µs scale) >= this value')
    parser.add_argument('--motor-max', type=float, default=MOTOR_CONTROL_FAIL,
                        help='Fail if actuator_motors control[i] >= this (0-1)')
    parser.add_argument('--yaw-tol-deg', type=float, default=12.0)
    parser.add_argument('--test', choices=('all', 'drift', 'saturation'), default='all')
    parser.add_argument('--json-out', default='analysis/output/phase1_grade.json')
    parser.add_argument('--meta', default='analysis/output/phase1_mission_meta.json',
                        help='Mission sidecar from run_sitl_mission.py')
    args = parser.parse_args(argv)

    if args.latest:
        log_dir = os.path.join(
            args.latest, 'build', 'px4_sitl_default', 'rootfs', 'log'
        )
        ulg_path = get_latest_log(log_dir)
    elif args.ulog:
        ulg_path = args.ulog
    else:
        parser.error('Provide ulog path or --latest PX4_ROOT')

    run_drift = args.test in ('all', 'drift')
    run_saturation = args.test in ('all', 'saturation')

    expect_home_m = None
    if os.environ.get('HOME_AMSL_M'):
        expect_home_m = float(os.environ['HOME_AMSL_M'])

    wind_speed = args.wind_speed
    wind_from = args.wind_from
    cruise = args.cruise_speed
    path_heading_deg = args.path_heading_deg
    mission_meta = None
    if os.path.isfile(args.meta):
        with open(args.meta, encoding='utf-8') as f:
            mission_meta = json.load(f)
        wind_speed = mission_meta.get('wind_speed_m_s', wind_speed)
        wind_from = mission_meta.get('wind_from_deg', wind_from)
        cruise = mission_meta.get('cruise_speed_m_s', cruise)
        if path_heading_deg is None and 'path_heading_deg' in mission_meta:
            path_heading_deg = mission_meta['path_heading_deg']
        print(f"Loaded mission meta from {args.meta}")
        if expect_home_m is not None:
            print(f"Highland gate: expect home_amsl_m={expect_home_m:.0f} m")

    report = grade_phase_1_sitl(
        ulg_path,
        wind_speed=wind_speed,
        wind_from_deg=wind_from,
        cruise_speed_m_s=cruise,
        path_heading_deg=path_heading_deg,
        max_xte_m=args.max_xte,
        pwm_fail_threshold=args.pwm_max,
        motor_control_fail=args.motor_max,
        yaw_tolerance_deg=args.yaw_tol_deg,
        run_drift=run_drift,
        run_saturation=run_saturation,
        mission_meta=mission_meta,
        expect_home_m=expect_home_m,
    )
    print_report(report)
    save_report(report, args.json_out)
    return 0 if report.passed else 1


if __name__ == '__main__':
    sys.exit(main())
