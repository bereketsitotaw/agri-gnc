from mavsdk.mission import MissionItem
from trajectory.waypoint import Waypoint
from geometry.utm_transform import UTMConverter


def convert_to_mission_items(
    trajectory: list[Waypoint],
    converter: UTMConverter
) -> list[MissionItem]:
    """
    Converts local metric waypoints back to WGS84 and packages them as MAVSDK MissionItems.
    Applies dynamic is_fly_through logic to prevent boundary overshoot.
    """
    mission_items = []

    for wp in trajectory:
        lon, lat = converter.utm_to_lonlat(wp.x, wp.y)

        # Bezier turnaround points must be flown through to maintain the smooth arc.
        # Sweep endpoints (is_spraying=True) must NOT be flown through to prevent field boundary overshoot.
        fly_through = not wp.is_spraying
        accept_radius = float('nan') if fly_through else 1.0

        item = MissionItem(
            latitude_deg=lat,
            longitude_deg=lon,
            relative_altitude_m=wp.z,
            speed_m_s=wp.velocity,
            is_fly_through=fly_through,
            gimbal_pitch_deg=float('nan'),
            gimbal_yaw_deg=float('nan'),
            camera_action=MissionItem.CameraAction.NONE,
            loiter_time_s=float('nan'),
            camera_photo_interval_s=float('nan'),
            acceptance_radius_m=accept_radius,
            yaw_deg=wp.yaw_deg if wp.yaw_deg is not None else float('nan'),
            camera_photo_distance_m=float('nan'),
            vehicle_action=MissionItem.VehicleAction.NONE,
        )
        mission_items.append(item)

    return mission_items
