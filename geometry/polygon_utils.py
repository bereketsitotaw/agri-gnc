from shapely.geometry import Polygon
from shapely.ops import orient
from shapely.validation import make_valid

from geometry.utm_transform import project_to_utm


def validate_and_clean_polygon(poly: Polygon) -> Polygon:
    """Returns a valid, counter-clockwise polygon suitable for coverage planning."""
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type != "Polygon":
        raise ValueError(f"Expected Polygon after cleaning, got {poly.geom_type}")
    return orient(poly, sign=1.0)
