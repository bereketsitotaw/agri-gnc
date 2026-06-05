from dataclasses import dataclass

from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform


@dataclass
class UTMConverter:
    """Bidirectional WGS84 (lon/lat) ↔ UTM converter for a fixed zone."""

    transformer_to_utm: Transformer
    transformer_to_wgs84: Transformer
    epsg: int

    def to_utm(self, lon: float, lat: float) -> tuple[float, float]:
        easting, northing = self.transformer_to_utm.transform(lon, lat)
        return easting, northing

    def to_wgs84(self, easting: float, northing: float) -> tuple[float, float]:
        lon, lat = self.transformer_to_wgs84.transform(easting, northing)
        return lon, lat

    def lonlat_to_utm(self, lon: float, lat: float) -> tuple[float, float]:
        return self.to_utm(lon, lat)

    def utm_to_lonlat(self, easting: float, northing: float) -> tuple[float, float]:
        return self.to_wgs84(easting, northing)


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def project_to_utm(poly: Polygon) -> tuple[Polygon, UTMConverter]:
    """
    Projects a WGS84 polygon (x=longitude, y=latitude) to UTM meters.
    Zone is chosen from the polygon centroid.
    """
    centroid = poly.centroid
    epsg = _utm_epsg(centroid.x, centroid.y)

    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs84 = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    converter = UTMConverter(
        transformer_to_utm=to_utm,
        transformer_to_wgs84=to_wgs84,
        epsg=epsg,
    )

    def _forward(x: float, y: float, z=None) -> tuple[float, float]:
        return to_utm.transform(x, y)

    utm_poly = transform(_forward, poly)
    return utm_poly, converter
