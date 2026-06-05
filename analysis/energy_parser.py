"""
ULog battery energy extraction for agri_gnc benchmark flights.

Integrates discharge power from battery_status and reports total energy [J, Wh].
For fair A/B tests (energy-optimal vs constant cruise), compare the same log
window — preferably while actuator_armed is true.

PX4 SITL often logs current_a as a constant negative placeholder; discharge
power is computed as |V * I| and SOC drop (remaining fraction) is reported
when available.
"""

import os
import sys

import numpy as np
from pyulog import ULog

from analysis.ulog_parser import get_latest_log


def _armed_mask(ulog: ULog, time_us: np.ndarray) -> np.ndarray | None:
    """Map battery timestamps to armed=True intervals from actuator_armed."""
    available = [d.name for d in ulog.data_list]
    if 'actuator_armed' not in available:
        return None

    aa = ulog.get_dataset('actuator_armed').data
    aa_t = aa['timestamp']
    aa_armed = aa['armed'].astype(bool)

    idx = np.searchsorted(aa_t, time_us, side='left')
    idx = np.clip(idx, 0, len(aa_t) - 1)
    idx_left = np.clip(idx - 1, 0, len(aa_t) - 1)
    dist_r = np.abs(aa_t[idx] - time_us)
    dist_l = np.abs(aa_t[idx_left] - time_us)
    closest = np.where(dist_l < dist_r, idx_left, idx)
    return aa_armed[closest]


def parse_battery_energy(ulog: ULog, armed_only: bool = True) -> dict:
    """Extract power and integrated discharge energy from battery_status."""
    available = [d.name for d in ulog.data_list]
    if 'battery_status' not in available:
        raise ValueError(
            f"ULog missing 'battery_status' topic. Available: {len(available)} topics"
        )

    data = ulog.get_dataset('battery_status').data
    time_us = np.asarray(data['timestamp'])
    time_s = (time_us - time_us[0]) / 1e6
    voltage = np.asarray(data['voltage_v'], dtype=float)
    current = np.asarray(data['current_a'], dtype=float)

    # PX4: negative current = discharging; SITL may hold a constant -1 A stub.
    power_w = np.abs(voltage * current)

    mask = np.ones(len(time_s), dtype=bool)
    if armed_only:
        armed = _armed_mask(ulog, time_us)
        if armed is not None:
            mask &= armed

    if not np.any(mask):
        mask[:] = True

    t = time_s[mask]
    p = power_w[mask]
    dt = np.diff(t)
    if len(dt) == 0:
        raise ValueError("battery_status has insufficient samples in selected window")

    energy_j = float(np.sum(p[:-1] * dt))
    duration_s = float(t[-1] - t[0])

    remaining = np.asarray(data['remaining'], dtype=float)
    soc_start = float(remaining[0])
    soc_end = float(remaining[-1])
    soc_consumed = max(0.0, soc_start - soc_end)

    current_unique = np.unique(np.round(current, 4))
    sitl_stub = len(current_unique) == 1

    return {
        'time_s': t,
        'power_w': p,
        'duration_s': duration_s,
        'energy_j': energy_j,
        'energy_wh': energy_j / 3600.0,
        'max_power_w': float(np.max(p)),
        'mean_power_w': float(np.mean(p)),
        'n_samples': int(len(t)),
        'soc_start': soc_start,
        'soc_end': soc_end,
        'soc_consumed_fraction': soc_consumed,
        'armed_only': armed_only,
        'sitl_current_stub': sitl_stub,
    }


def print_energy_summary(stats: dict, ulg_path: str) -> None:
    window = "armed flight only" if stats['armed_only'] else "full log"
    print("\n" + "=" * 50)
    print("  ENERGY CONSUMPTION SUMMARY")
    print("=" * 50)
    print(f"  Log file:         {os.path.basename(ulg_path)}")
    print(f"  Window:           {window}")
    print(f"  Samples:          {stats['n_samples']:,}")
    print(f"  Duration:         {stats['duration_s']:.1f} s")
    print(f"  Max power draw:   {stats['max_power_w']:.1f} W")
    print(f"  Average power:    {stats['mean_power_w']:.1f} W")
    print(f"  Total energy:     {stats['energy_j']:.1f} J ({stats['energy_wh']:.2f} Wh)")
    print(f"  SOC consumed:     {stats['soc_consumed_fraction'] * 100:.2f}% "
          f"({stats['soc_start']:.3f} → {stats['soc_end']:.3f})")
    if stats['sitl_current_stub']:
        print("  Note:             constant current_a — typical PX4 SITL stub;")
        print("                    prefer SOC delta or relative |V*I| vs baseline flight")
    print("=" * 50 + "\n")


def analyze_energy(ulg_path: str, save_csv: bool = True) -> dict:
    print(f"Parsing energy from: {ulg_path}")
    ulog = ULog(ulg_path)
    stats = parse_battery_energy(ulog, armed_only=True)
    print_energy_summary(stats, ulg_path)

    if save_csv:
        os.makedirs('analysis/output', exist_ok=True)
        out = 'analysis/output/energy_data.csv'
        np.savetxt(
            out,
            np.column_stack([stats['time_s'], stats['power_w']]),
            delimiter=',',
            header='time_s,power_w',
            comments='',
        )
        print(f"  → Saved power time-series to {out}")

    return stats


def analyze_latest_energy(px4_root_dir: str) -> dict:
    log_dir = os.path.join(px4_root_dir, 'build', 'px4_sitl_default', 'rootfs', 'log')
    print("Searching for latest flight log...")
    latest_log = get_latest_log(log_dir)
    print(f"Found: {latest_log}")
    return analyze_energy(latest_log)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_energy(sys.argv[1])
    else:
        PX4_DIR = "/Users/berekets.kidane/Desktop/PHD/Year2/Projects/PX4-Autopilot"
        analyze_latest_energy(PX4_DIR)
