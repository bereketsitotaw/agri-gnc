import numpy as np
from pyproj import Transformer

from mavsdk.mission import MissionItem
from trajectory.waypoint import Waypoint
from geometry.utm_transform import UTMConverter, _utm_epsg
from px4.mavlink_converter import convert_to_mission_items


def _converter_at(lon: float, lat: float) -> UTMConverter:
    """UTMConverter for a fixed WGS84 origin (zone from lon/lat)."""
    epsg = _utm_epsg(lon, lat)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    to_wgs84 = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    converter = UTMConverter(
        transformer_to_utm=to_utm,
        transformer_to_wgs84=to_wgs84,
        epsg=epsg,
    )
    return converter


def test_conversion_preserves_wgs84_invariants():
    converter = _converter_at(-97.1000, 32.7300)
    origin_x, origin_y = converter.lonlat_to_utm(-97.1000, 32.7300)

    waypoint = Waypoint(
        x=origin_x,
        y=origin_y,
        z=15.0,
        velocity=8.0,
        is_spraying=True,
    )
    mission_items = convert_to_mission_items([waypoint], converter)

    assert len(mission_items) == 1
    assert abs(mission_items[0].longitude_deg - (-97.1000)) < 1e-5
    assert abs(mission_items[0].latitude_deg - 32.7300) < 1e-5
    assert mission_items[0].relative_altitude_m == 15.0


def test_mission_item_flight_dynamics():
    converter = _converter_at(-97.1000, 32.7300)
    origin_x, origin_y = converter.lonlat_to_utm(-97.1000, 32.7300)

    waypoints = [
        Waypoint(
            x=origin_x, y=origin_y, z=10.0, velocity=8.0, is_spraying=True
        ),
        Waypoint(
            x=origin_x + 10.0, y=origin_y, z=10.0, velocity=4.0, is_spraying=False
        ),
    ]
    mission_items = convert_to_mission_items(waypoints, converter)

    assert len(mission_items) == 2
    assert mission_items[0].is_fly_through is False
    assert mission_items[0].acceptance_radius_m == 1.0
    assert mission_items[1].is_fly_through is True
    assert np.isnan(mission_items[1].acceptance_radius_m)
    assert mission_items[0].speed_m_s == 8.0
    assert mission_items[1].speed_m_s == 4.0
