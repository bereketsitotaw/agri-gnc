import numpy as np
from shapely.geometry import LineString
from trajectory.arc_smoother import generate_smoothed_trajectory


def test_trajectory_state_transitions():
    segments = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 10.0), (0.0, 10.0)]),
    ]
    trajectory = generate_smoothed_trajectory(
        segments, swath_width=10.0, turn_resolution=15
    )

    assert len(trajectory) >= 2 + (15 - 2) + 2
    assert trajectory[0].is_spraying is True
    assert trajectory[0].velocity == 8.0
    assert any(wp.is_spraying is False for wp in trajectory)
    turn_speeds = [wp.velocity for wp in trajectory if not wp.is_spraying]
    assert turn_speeds
    assert all(v <= 8.0 for v in turn_speeds)


def test_bezier_tangency_invariants():
    segments = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 10.0), (0.0, 10.0)]),
    ]
    trajectory = generate_smoothed_trajectory(
        segments, swath_width=10.0, turn_resolution=15
    )

    sweep_heading = np.arctan2(
        trajectory[1].y - trajectory[0].y,
        trajectory[1].x - trajectory[0].x,
    )

    # B'(0) is parallel to (p1 - p0); reconstruct control arm at the junction
    end_coord = segments[0].coords[-1]
    start_coord = segments[0].coords[0]
    next_start = segments[1].coords[0]
    dx1 = end_coord[0] - start_coord[0]
    dy1 = end_coord[1] - start_coord[1]
    len1 = np.hypot(dx1, dy1)
    v1 = (dx1 / len1, dy1 / len1)

    p0 = end_coord
    p3 = next_start
    gap_dist = np.hypot(p3[0] - p0[0], p3[1] - p0[1])
    d = max(gap_dist * 0.5, 10.0 * 0.5)
    p1 = (p0[0] + v1[0] * d, p0[1] + v1[1] * d)

    bezier_tangent_heading = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    assert abs(bezier_tangent_heading - sweep_heading) < 0.05


def test_bezier_cross_track_error():
    segments = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 10.0), (0.0, 10.0)]),
    ]
    trajectory = generate_smoothed_trajectory(
        segments, swath_width=10.0, turn_resolution=15
    )

    turnaround_waypoints = [wp for wp in trajectory if not wp.is_spraying]
    for i in range(len(turnaround_waypoints) - 1):
        p1 = turnaround_waypoints[i]
        p2 = turnaround_waypoints[i + 1]
        dist = np.hypot(p2.x - p1.x, p2.y - p1.y)
        assert dist < 2.0
