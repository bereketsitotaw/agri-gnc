"""
Phase I Sprint 3 — smallholder spray geofencing (TDD).

Acceptance: "The Dry Corridor" — one eastbound leg at y=50 across two plots
separated by a 20 m dry gap must decompose into Spray ON / OFF / ON segments.
"""

import pytest
from shapely.geometry import LineString, Polygon

from shapely.geometry import LineString

from geometry.spray_geofence import SprayGeofence, SprayLegSlice
from trajectory.arc_smoother import generate_smoothed_trajectory


def _dry_corridor_fence() -> SprayGeofence:
    plot_alpha = Polygon([(0, 0), (50, 0), (50, 100), (0, 100)])
    plot_beta = Polygon([(70, 0), (120, 0), (120, 100), (70, 100)])
    return SprayGeofence([plot_alpha, plot_beta])


def test_dry_corridor_three_phase_decomposition():
    """Single sweep x: 0→120 at y=50 → ON [0,50], OFF [50,70], ON [70,120]."""
    fence = _dry_corridor_fence()
    slices = fence.slice_spray_leg((0.0, 50.0), (120.0, 50.0))

    assert len(slices) == 3
    assert slices[0].is_spraying is True
    assert slices[1].is_spraying is False
    assert slices[2].is_spraying is True

    assert slices[0].start == pytest.approx((0.0, 50.0))
    assert slices[0].end == pytest.approx((50.0, 50.0))
    assert slices[1].start == pytest.approx((50.0, 50.0))
    assert slices[1].end == pytest.approx((70.0, 50.0))
    assert slices[2].start == pytest.approx((70.0, 50.0))
    assert slices[2].end == pytest.approx((120.0, 50.0))


def test_dry_corridor_segment_lengths():
    fence = _dry_corridor_fence()
    slices = fence.slice_spray_leg((0.0, 50.0), (120.0, 50.0))

    lengths = [
        LineString([s.start, s.end]).length for s in slices
    ]
    assert lengths == pytest.approx([50.0, 20.0, 50.0])


def test_leg_fully_inside_single_plot():
    plot = Polygon([(0, 0), (50, 0), (50, 100), (0, 100)])
    fence = SprayGeofence([plot])
    slices = fence.slice_spray_leg((5.0, 50.0), (45.0, 50.0))

    assert len(slices) == 1
    assert slices[0].is_spraying is True
    assert slices[0].start == pytest.approx((5.0, 50.0))
    assert slices[0].end == pytest.approx((45.0, 50.0))


def test_leg_fully_in_dry_corridor_only():
    fence = _dry_corridor_fence()
    slices = fence.slice_spray_leg((52.0, 50.0), (68.0, 50.0))

    assert len(slices) == 1
    assert slices[0].is_spraying is False
    assert slices[0].start == pytest.approx((52.0, 50.0))
    assert slices[0].end == pytest.approx((68.0, 50.0))


def test_reverse_leg_preserves_order_along_track():
    """Westbound leg must still list segments from start_wp → end_wp."""
    fence = _dry_corridor_fence()
    slices = fence.slice_spray_leg((120.0, 50.0), (0.0, 50.0))

    assert len(slices) == 3
    assert slices[0].is_spraying is True
    assert slices[0].start == pytest.approx((120.0, 50.0))
    assert slices[0].end == pytest.approx((70.0, 50.0))
    assert slices[1].is_spraying is False
    assert slices[2].is_spraying is True
    assert slices[2].end == pytest.approx((0.0, 50.0))


def test_empty_plot_list_raises():
    with pytest.raises(ValueError, match="at least one"):
        SprayGeofence([])


def test_slices_are_dataclass_instances():
    fence = _dry_corridor_fence()
    slices = fence.slice_spray_leg((0.0, 50.0), (120.0, 50.0))
    assert all(isinstance(s, SprayLegSlice) for s in slices)


def test_trajectory_dry_corridor_transit_waypoints():
    """One eastbound sweep across two plots → transit waypoints in the gap."""
    fence = _dry_corridor_fence()
    segments = [LineString([(0.0, 50.0), (120.0, 50.0)])]
    trajectory = generate_smoothed_trajectory(
        segments,
        swath_width=30.0,
        turn_resolution=5,
        wind_speed=0.0,
        enable_drift_compensation=False,
        geofence=fence,
    )

    assert any(wp.is_spraying for wp in trajectory)
    assert any(not wp.is_spraying for wp in trajectory)

    corridor_x = [wp.x for wp in trajectory if not wp.is_spraying]
    assert corridor_x
    assert all(50.0 <= x <= 70.0 for x in corridor_x)
