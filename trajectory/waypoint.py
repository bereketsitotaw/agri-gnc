from dataclasses import dataclass


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    velocity: float
    is_spraying: bool
    yaw_deg: float | None = None  # MAVLink yaw (crab); None → autopilot holds course
