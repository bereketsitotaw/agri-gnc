"""Sprint 3 — planned mission has zero off-plot spray intent."""

from shapely.geometry import LineString, Point, Polygon

from cpp.sweep_optimizer import optimize_sweep_angle
from geometry.dry_corridor_demo import hawassa_dry_corridor_geometry
from geometry.polygon_utils import validate_and_clean_polygon, project_to_utm
from trajectory.arc_smoother import generate_smoothed_trajectory

from analysis.grade_sprint_3 import _load_active_zone, _in_corridor


def _demo_trajectory_meta():
    anchor = Polygon([
        (38.47, 7.05), (38.471, 7.05), (38.471, 7.051), (38.47, 7.051),
    ])
    _, converter = project_to_utm(validate_and_clean_polygon(anchor))
    poly, geofence, meta = hawassa_dry_corridor_geometry(converter)
    _, segments, _ = optimize_sweep_angle(poly, 30.0, 5.0)
    traj = generate_smoothed_trajectory(
        segments, 30.0, turn_resolution=5, wind_speed=0.0,
        enable_drift_compensation=False, geofence=geofence,
    )
    meta['planned_waypoints_utm'] = [
        {'x': wp.x, 'y': wp.y, 'is_spraying': wp.is_spraying} for wp in traj
    ]
    return meta


def test_planned_spray_waypoints_inside_active_zone():
    meta = _demo_trajectory_meta()
    zone = _load_active_zone(meta)

    for wp in meta['planned_waypoints_utm']:
        if not wp['is_spraying']:
            continue
        assert zone.buffer(1e-3).contains(Point(wp['x'], wp['y']))


def test_planned_corridor_waypoints_are_transit():
    meta = _demo_trajectory_meta()

    for wp in meta['planned_waypoints_utm']:
        if _in_corridor(wp['x'], meta):
            assert wp['is_spraying'] is False
