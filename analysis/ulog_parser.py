"""
ULog Telemetry Extraction & Cross-Track Error Benchmarking Pipeline
===================================================================
Core analysis module for agri_gnc PhD research.

Extracts actual and commanded flight paths from PX4 .ulg logs,
computes Cross-Track Error (XTE), and generates publication-quality plots.
"""

import os
import glob
from pyulog import ULog
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Log Discovery
# ---------------------------------------------------------------------------

def get_latest_log(log_dir):
    """Finds the most recent .ulg file in the PX4 log directory."""
    search_path = os.path.join(log_dir, '**', '*.ulg')
    list_of_files = glob.glob(search_path, recursive=True)
    if not list_of_files:
        raise FileNotFoundError(f"No .ulg files found in {log_dir}")
    return max(list_of_files, key=os.path.getctime)


# ---------------------------------------------------------------------------
# Data Extraction
# ---------------------------------------------------------------------------

def parse_local_position(ulog):
    """Extracts the actual flown path (vehicle_local_position) in NED frame."""
    data = ulog.get_dataset('vehicle_local_position')

    df = pd.DataFrame({
        'timestamp': data.data['timestamp'],
        'x': data.data['x'],  # North
        'y': data.data['y'],  # East
        'z': data.data['z'],  # Down
    })

    # Convert timestamps from microseconds to seconds, zero-based
    df['time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e6
    return df


def parse_position_setpoint(ulog):
    """Extracts the commanded reference path (vehicle_local_position_setpoint)."""
    data = ulog.get_dataset('vehicle_local_position_setpoint')

    df = pd.DataFrame({
        'timestamp': data.data['timestamp'],
        'x': data.data['x'],  # Commanded North
        'y': data.data['y'],  # Commanded East
        'z': data.data['z'],  # Commanded Down
    })

    df['time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e6
    return df


# ---------------------------------------------------------------------------
# Cross-Track Error Computation
# ---------------------------------------------------------------------------

def compute_xte(actual_df, setpoint_df):
    """
    Computes Cross-Track Error by time-aligning actual vs. commanded positions.

    For each actual position sample, finds the nearest-in-time setpoint and
    computes the Euclidean distance in the horizontal (North-East) plane.

    Returns a DataFrame with columns: time_s, xte_m, xte_north, xte_east
    """
    # Use the actual timestamps as the reference timeline
    actual_times = actual_df['timestamp'].values
    sp_times = setpoint_df['timestamp'].values

    # For each actual sample, find the index of the closest setpoint in time
    indices = np.searchsorted(sp_times, actual_times, side='left')
    indices = np.clip(indices, 0, len(sp_times) - 1)

    # Also check index-1 to find the truly closest sample
    idx_left = np.clip(indices - 1, 0, len(sp_times) - 1)
    dist_right = np.abs(sp_times[indices] - actual_times)
    dist_left = np.abs(sp_times[idx_left] - actual_times)
    closest = np.where(dist_left < dist_right, idx_left, indices)

    # Compute per-axis and total horizontal error
    dx = actual_df['x'].values - setpoint_df['x'].values[closest]  # North error
    dy = actual_df['y'].values - setpoint_df['y'].values[closest]  # East error
    xte = np.sqrt(dx**2 + dy**2)

    return pd.DataFrame({
        'time_s': actual_df['time_s'].values,
        'xte_m': xte,
        'xte_north': dx,
        'xte_east': dy,
    })


# ---------------------------------------------------------------------------
# Visualization — Publication Quality
# ---------------------------------------------------------------------------

def plot_trajectory_comparison(actual_df, setpoint_df, output_path):
    """Plots actual vs. commanded 2D trajectory in the NED horizontal plane."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(setpoint_df['y'], setpoint_df['x'],
            label='Commanded Path', color='#e74c3c', linewidth=2,
            linestyle='--', alpha=0.8)
    ax.plot(actual_df['y'], actual_df['x'],
            label='Actual Flight Path', color='#2980b9', linewidth=1.5)

    ax.set_title('Trajectory Comparison (Local NED Frame)', fontsize=14, fontweight='bold')
    ax.set_xlabel('East (meters)', fontsize=12)
    ax.set_ylabel('North (meters)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(fontsize=11)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    print(f"  → Saved trajectory plot to {output_path}")
    plt.close(fig)


def plot_3d_trajectory(pos_df, output_path):
    """Plots actual 3D trajectory in the NED frame."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Invert Z to show altitude (NED frame means +Z is down)
    altitude = -pos_df['z']
    
    ax.plot(pos_df['y'], pos_df['x'], altitude, label='3D Flight Path', color='green', linewidth=2)
    ax.set_title('3D Flown Trajectory (Local NED Frame)', fontsize=14, fontweight='bold')
    ax.set_xlabel('East (meters)', fontsize=12)
    ax.set_ylabel('North (meters)', fontsize=12)
    ax.set_zlabel('Altitude (meters)')
    ax.legend(fontsize=11)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  → Saved 3D flight path visualization to {output_path}")
    plt.close(fig)



def plot_xte_timeseries(xte_df, output_path):
    """Plots Cross-Track Error over time with summary statistics."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(xte_df['time_s'], xte_df['xte_m'],
            color='#2980b9', linewidth=0.8, alpha=0.7, label='XTE')

    # Summary statistics
    mean_xte = xte_df['xte_m'].mean()
    max_xte = xte_df['xte_m'].max()
    p95_xte = xte_df['xte_m'].quantile(0.95)
    rmse_xte = np.sqrt(np.mean(xte_df['xte_m']**2))

    ax.axhline(mean_xte, color='#e74c3c', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_xte:.3f} m')
    ax.axhline(p95_xte, color='#f39c12', linestyle=':', linewidth=1.5,
               label=f'95th %%ile: {p95_xte:.3f} m')

    ax.set_title('Cross-Track Error Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('XTE (meters)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')

    # Text box with key stats
    stats_text = (
        f"RMSE:  {rmse_xte:.4f} m\n"
        f"Mean:  {mean_xte:.4f} m\n"
        f"Max:   {max_xte:.4f} m\n"
        f"P95:   {p95_xte:.4f} m"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  → Saved XTE time-series to {output_path}")
    plt.close(fig)


def print_xte_summary(xte_df):
    """Prints a formatted summary of Cross-Track Error statistics."""
    xte = xte_df['xte_m']
    rmse = np.sqrt(np.mean(xte**2))

    print("\n" + "=" * 50)
    print("  CROSS-TRACK ERROR SUMMARY")
    print("=" * 50)
    print(f"  Samples:       {len(xte):,}")
    print(f"  Duration:      {xte_df['time_s'].iloc[-1]:.1f} s")
    print(f"  RMSE:          {rmse:.4f} m")
    print(f"  Mean:          {xte.mean():.4f} m")
    print(f"  Std Dev:       {xte.std():.4f} m")
    print(f"  Median:        {xte.median():.4f} m")
    print(f"  Max:           {xte.max():.4f} m")
    print(f"  95th %%ile:    {xte.quantile(0.95):.4f} m")
    print(f"  99th %%ile:    {xte.quantile(0.99):.4f} m")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Main Analysis Entry Point
# ---------------------------------------------------------------------------

def analyze_flight(ulg_path):
    """
    Full analysis pipeline for a single .ulg file.

    Args:
        ulg_path: Absolute path to a .ulg file.
    """
    print(f"Parsing ULog: {ulg_path}")
    ulog = ULog(ulg_path)

    # List available topics for diagnostics
    available = [d.name for d in ulog.data_list]
    print(f"  Available topics: {len(available)}")

    # --- Extract actual position ---
    if 'vehicle_local_position' not in available:
        raise ValueError("ULog missing 'vehicle_local_position' topic")
    actual_df = parse_local_position(ulog)
    print(f"  Actual position:  {len(actual_df):,} samples")

    # --- Extract commanded setpoint ---
    has_setpoint = 'vehicle_local_position_setpoint' in available
    if has_setpoint:
        setpoint_df = parse_position_setpoint(ulog)
        print(f"  Setpoint:         {len(setpoint_df):,} samples")
    else:
        print("  ⚠ No 'vehicle_local_position_setpoint' found — skipping XTE")
        setpoint_df = None

    # --- Ensure output directory exists ---
    os.makedirs('analysis/output', exist_ok=True)

    # --- Plot trajectory ---
    if setpoint_df is not None:
        plot_trajectory_comparison(
            actual_df, setpoint_df,
            'analysis/output/trajectory_comparison.png'
        )
        
        plot_3d_trajectory(actual_df, 'analysis/output/latest_flight_path_3d.png')


        # --- Compute and plot XTE ---
        xte_df = compute_xte(actual_df, setpoint_df)
        print_xte_summary(xte_df)
        plot_xte_timeseries(xte_df, 'analysis/output/xte_timeseries.png')

        # --- Save raw data to CSV for further analysis ---
        xte_df.to_csv('analysis/output/xte_data.csv', index=False)
        print("  → Saved raw XTE data to analysis/output/xte_data.csv")
    else:
        # Plot just the actual path
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(actual_df['y'], actual_df['x'],
                label='Actual Flight Path', color='#2980b9', linewidth=1.5)
        ax.set_title('Actual Flown Trajectory (Local NED Frame)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('East (meters)', fontsize=12)
        ax.set_ylabel('North (meters)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend()
        fig.tight_layout()
        fig.savefig('analysis/output/actual_trajectory.png', dpi=150)
        print("  → Saved actual trajectory to analysis/output/actual_trajectory.png")
        plt.close(fig)
        
        plot_3d_trajectory(actual_df, 'analysis/output/latest_flight_path_3d.png')


    print("\nAnalysis complete.")


def analyze_latest_flight(px4_root_dir):
    """Finds and analyzes the most recent flight log from a PX4 build."""
    log_dir = os.path.join(px4_root_dir, 'build', 'px4_sitl_default', 'rootfs', 'log')
    print("Searching for latest flight log...")
    latest_log = get_latest_log(log_dir)
    print(f"Found: {latest_log}")
    analyze_flight(latest_log)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Direct path to a .ulg file
        analyze_flight(sys.argv[1])
    else:
        # Default: search PX4 build directory
        PX4_DIR = "/Users/berekets.kidane/Desktop/PHD/Year2/Projects/PX4-Autopilot"
        analyze_latest_flight(PX4_DIR)
