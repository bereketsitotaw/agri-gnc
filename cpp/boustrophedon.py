import numpy as np
from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.affinity import rotate


def rotate_polygon(poly: Polygon, angle_deg: float, origin: str = 'centroid') -> Polygon:
    """Rotates a polygon by angle_deg. Origin defaults to centroid to prevent drift."""
    return rotate(poly, angle_deg, origin=origin, use_radians=False)


def generate_sweep_lines(poly: Polygon, swath_width: float, sweep_angle_deg: float) -> list[LineString]:
    if swath_width <= 0:
        raise ValueError(f"Swath width must be > 0, got {swath_width}")

    # 1. Rotate polygon to align sweeps horizontally
    rotated_poly = rotate_polygon(poly, -sweep_angle_deg)
    minx, miny, maxx, maxy = rotated_poly.bounds

    # 2. Generate horizontal sweep lines
    y_coords = np.arange(miny + swath_width/2.0, maxy, swath_width)
    segments = []

    for y in y_coords:
        line = LineString([(minx - 10.0, y), (maxx + 10.0, y)])
        intersection = rotated_poly.intersection(line)

        # 3. Handle intersections rigorously
        if intersection.is_empty:
            continue
        elif intersection.geom_type == 'LineString':
            segments.append(intersection)
        elif intersection.geom_type == 'MultiLineString':
            # Flatten disconnected segments (e.g., across concave gaps)
            segments.extend(list(intersection.geoms))
        else:
            raise ValueError(f"Unexpected intersection geometry type: {intersection.geom_type}")

    # 4. Sort segments by Y (bottom-to-top), then X (left-to-right) to ensure deterministic ordering
    segments.sort(key=lambda line: (line.coords[0][1], line.coords[0][0]))

    # 5. Rotate segments back to the original orientation
    final_segments = []
    rot_centroid = rotated_poly.centroid
    for seg in segments:
        restored_seg = rotate(seg, sweep_angle_deg, origin=rot_centroid, use_radians=False)
        final_segments.append(restored_seg)

    return final_segments


def order_sweeps_boustrophedon(segments: list[LineString]) -> list[LineString]:
    """
    Alternates segment directions to create a continuous flyable path.
    Reverses the coordinates of every odd-indexed segment.
    """
    ordered = []
    for i, seg in enumerate(segments):
        coords = list(seg.coords)
        if i % 2 != 0:
            coords.reverse()
        ordered.append(LineString(coords))
    return ordered
