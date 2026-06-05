"""
Hawassa Sprint 3 dry-corridor demo geometry (UTM meters).

Plot A: 100×200 m, 20 m gap, Plot B: 100×200 m — envelope 220×200 m for CPP.
"""

from __future__ import annotations

from shapely.geometry import Polygon

import math

from geometry.spray_geofence import SprayGeofence
from geometry.utm_transform import UTMConverter
from trajectory.drift_model import SprayDriftParams, drift_displacement_enu


def hawassa_dry_corridor_geometry(
    converter: UTMConverter,
    home_lon: float = 38.470000,
    home_lat: float = 7.050000,
) -> tuple[Polygon, SprayGeofence, dict]:
    """
    Build planning envelope and geofence in UTM anchored at Hawassa home.

    Returns (planning_envelope_utm, geofence, meta_dict).
    """
    e0, n0 = converter.lonlat_to_utm(home_lon, home_lat)

    plot_a = Polygon([
        (e0, n0),
        (e0 + 100.0, n0),
        (e0 + 100.0, n0 + 200.0),
        (e0, n0 + 200.0),
    ])
    plot_b = Polygon([
        (e0 + 120.0, n0),
        (e0 + 220.0, n0),
        (e0 + 220.0, n0 + 200.0),
        (e0 + 120.0, n0 + 200.0),
    ])
    envelope = Polygon([
        (e0, n0),
        (e0 + 220.0, n0),
        (e0 + 220.0, n0 + 200.0),
        (e0, n0 + 200.0),
    ])
    geofence = SprayGeofence([plot_a, plot_b])

    # MoA 1.0 m kinematic margin + bounded SITL drift envelope when compensation is active
    drift_dx, drift_dy = drift_displacement_enu(
        5.0, 180.0, SprayDriftParams(altitude_m=15.0, nozzle_micron=300),
    )
    drift_envelope_m = min(math.hypot(drift_dx, drift_dy), 4.0)

    meta = {
        'demo': 'hawassa_dry_corridor',
        'home_utm_e': e0,
        'home_utm_n': n0,
        'home_lon': home_lon,
        'home_lat': home_lat,
        'plot_a_utm': list(plot_a.exterior.coords[:-1]),
        'plot_b_utm': list(plot_b.exterior.coords[:-1]),
        'corridor_x_utm': [e0 + 100.0, e0 + 120.0],
        'overspray_tolerance_m': 1.0 + drift_envelope_m,
        'kinematic_tolerance_m': 1.0,
        'drift_envelope_m': drift_envelope_m,
    }
    return envelope, geofence, meta
