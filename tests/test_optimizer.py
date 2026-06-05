from shapely.geometry import Polygon, LineString

from cpp.coverage_metrics import compute_path_metrics
from cpp.sweep_optimizer import optimize_sweep_angle


def test_compute_path_metrics():
    segments = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 5.0), (0.0, 5.0)]),
    ]
    metrics = compute_path_metrics(segments)

    assert metrics["turn_count"] == 1
    assert metrics["total_length"] == 25.0


def test_optimize_sweep_angle_picks_longest_edge():
    rect = Polygon([(0, 0), (200, 0), (200, 50), (0, 50)])
    best_angle, _, best_metrics = optimize_sweep_angle(
        rect, swath_width=10.0, angle_step=15.0
    )

    assert best_angle < 15.0
    assert best_metrics["turn_count"] <= 5
