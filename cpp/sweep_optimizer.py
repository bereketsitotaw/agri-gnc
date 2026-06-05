import numpy as np
import warnings
from shapely.errors import TopologicalError
from shapely.geometry import Polygon, LineString
from cpp.boustrophedon import generate_sweep_lines, order_sweeps_boustrophedon
from cpp.coverage_metrics import compute_path_metrics


def optimize_sweep_angle(
    poly: Polygon,
    swath_width: float,
    angle_step: float = 10.0,
    turn_penalty: float = 50.0
) -> tuple[float, list[LineString], dict]:
    """
    Searches angles [0, 180) by angle_step to find the optimal sweep orientation.
    Cost = (turn_count * turn_penalty) + total_length.

    Note: turn_penalty implicitly equates one turn to X meters of flight
    (e.g., 50m = ~6 seconds at 8 m/s cruise). Tune based on specific vehicle kinetics.

    Returns: (best_angle, best_ordered_segments, metrics_of_best)
    """
    angles = np.arange(0.0, 180.0, angle_step)

    best_angle = 0.0
    best_cost = float('inf')
    best_segments = []
    best_metrics = {}

    for angle in angles:
        try:
            # Generate and order
            raw_segments = generate_sweep_lines(poly, swath_width, angle)
            if not raw_segments:
                continue

            ordered = order_sweeps_boustrophedon(raw_segments)

            # Compute metrics and cost
            metrics = compute_path_metrics(ordered)
            cost = (metrics["turn_count"] * turn_penalty) + metrics["total_length"]

            # Update best
            if cost < best_cost:
                best_cost = cost
                best_angle = angle
                best_segments = ordered
                best_metrics = metrics
        except (ValueError, TopologicalError) as e:
            # Catch specific geometric failures without swallowing broader logic bugs
            warnings.warn(f"Skipping angle {angle:.1f}°: {e}")
            continue

    if best_cost == float('inf'):
        raise ValueError("Optimization failed: Could not generate valid sweeps at any angle.")

    return best_angle, best_segments, best_metrics
