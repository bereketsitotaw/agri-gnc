import pytest
from shapely.geometry import Polygon, LineString

from cpp.boustrophedon import (
    rotate_polygon,
    generate_sweep_lines,
    order_sweeps_boustrophedon,
)


def test_rotation_invariants():
    square = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    original_area = square.area
    original_centroid = square.centroid

    rotated = rotate_polygon(square, 45.0)

    assert abs(rotated.area - original_area) < 1e-6
    assert abs(rotated.centroid.x - original_centroid.x) < 1e-6
    assert abs(rotated.centroid.y - original_centroid.y) < 1e-6


def test_sweep_spacing():
    square = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    segments = generate_sweep_lines(square, swath_width=10.0, sweep_angle_deg=0.0)

    assert len(segments) == 10

    y0 = segments[0].coords[0][1]
    y1 = segments[1].coords[0][1]
    assert abs(y1 - y0) == 10.0


def test_boustrophedon_ordering():
    segments = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(0.0, 10.0), (10.0, 10.0)]),
        LineString([(0.0, 20.0), (10.0, 20.0)]),
    ]
    ordered = order_sweeps_boustrophedon(segments)

    assert ordered[0].coords[0] == (0.0, 0.0)
    assert ordered[1].coords[0] == (10.0, 10.0)
    assert ordered[2].coords[0] == (0.0, 20.0)


def test_concave_sweep_handling():
    u_shape = Polygon([
        (0, 0), (30, 0), (30, 30), (20, 30), (20, 10),
        (10, 10), (10, 30), (0, 30),
    ])
    segments = generate_sweep_lines(u_shape, swath_width=5.0, sweep_angle_deg=0.0)

    for seg in segments:
        assert seg.geom_type == 'LineString'

    assert len(segments) > 6
