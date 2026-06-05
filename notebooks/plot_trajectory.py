import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np

# Import our pipeline layers
from geometry.polygon_utils import validate_and_clean_polygon, project_to_utm
from cpp.sweep_optimizer import optimize_sweep_angle
from trajectory.arc_smoother import generate_smoothed_trajectory


def run_and_visualize():
    # 1. Define a WGS84 test farm in Texas (L-shape, approx 100m x 100m)
    # Longitude, Latitude
    wgs84_coords = [
        (-97.10000, 32.73000),  # SW Corner
        (-97.09893, 32.73000),  # SE Corner (Extended)
        (-97.09893, 32.73045),  # Inner corner East
        (-97.09957, 32.73045),  # Inner corner notch
        (-97.09957, 32.73090),  # NE Corner
        (-97.10000, 32.73090)   # NW Corner
    ]
    poly_wgs84 = Polygon(wgs84_coords)

    # 2. Geometry Layer: Clean and Project to UTM
    poly_valid = validate_and_clean_polygon(poly_wgs84)
    poly_utm, converter = project_to_utm(poly_valid)

    # 3. CPP Layer: Optimize sweeps (Swath = 10m)
    swath_width = 10.0
    best_angle, segments, metrics = optimize_sweep_angle(
        poly=poly_utm,
        swath_width=swath_width,
        angle_step=5.0
    )

    # 4. Trajectory Layer: Generate smooth trajectory
    trajectory = generate_smoothed_trajectory(
        segments=segments,
        swath_width=swath_width,
        flight_altitude=10.0,
        cruise_speed=8.0,
        turn_speed=4.0,
        turn_resolution=15
    )

    # Enhanced Logging
    print(f"Optimal Angle: {best_angle}° | Turns: {metrics['turn_count']} | "
          f"Path length: {metrics['total_length']:.1f}m | Waypoints: {len(trajectory)}")

    # 5. Plotting (in UTM metric space)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot field boundary
    x_bnd, y_bnd = poly_utm.exterior.xy
    ax.plot(x_bnd, y_bnd, color='black', linewidth=2, label='Field Boundary (UTM)')
    ax.fill(x_bnd, y_bnd, color='lightgreen', alpha=0.2)

    # Plot trajectory segments
    for i in range(len(trajectory) - 1):
        wp1 = trajectory[i]
        wp2 = trajectory[i + 1]

        # Note: Transitions from spray (True) to turn (False) will draw as red dashed.
        # This means the red curves start exactly at the boundary of the sweep endpoints.
        if wp1.is_spraying and wp2.is_spraying:
            ax.plot([wp1.x, wp2.x], [wp1.y, wp2.y], color='blue', linewidth=2, zorder=3)
        else:
            ax.plot([wp1.x, wp2.x], [wp1.y, wp2.y], color='red', linewidth=2, linestyle='--', zorder=2)

    # Plot start and end points
    ax.scatter(trajectory[0].x, trajectory[0].y, color='green', s=100, marker='^', label='Start', zorder=4)
    ax.scatter(trajectory[-1].x, trajectory[-1].y, color='red', s=100, marker='s', label='End', zorder=4)

    # Fake lines for the legend
    ax.plot([], [], color='blue', linewidth=2, label='Spraying (Cruise: 8 m/s)')
    ax.plot([], [], color='red', linewidth=2, linestyle='--', label='Transit (Turn: 4 m/s)')

    ax.set_aspect('equal')
    ax.set_title("End-to-End Pipeline Path (WGS84 -> UTM -> Trajectory)", fontweight='bold')
    ax.set_xlabel("Easting (meters)")
    ax.set_ylabel("Northing (meters)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_and_visualize()
