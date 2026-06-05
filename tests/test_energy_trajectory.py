from shapely.geometry import LineString

from trajectory.arc_smoother import cruise_speeds_for_sweeps, generate_smoothed_trajectory


def test_energy_optimal_speeds_decrease_with_spray_depletion():
    segments = [
        LineString([(0, 0), (200, 0)]),
        LineString([(200, 30), (0, 30)]),
        LineString([(0, 60), (200, 60)]),
    ]
    speeds = cruise_speeds_for_sweeps(
        segments, swath_width=30.0, use_energy_optimal=True
    )
    assert len(speeds) == 3
    assert speeds[0] > speeds[-1]


def test_trajectory_waypoint_speeds_reflect_energy_profile():
    segments = [LineString([(0, 0), (100, 0)]), LineString([(100, 10), (0, 10)])]
    speeds = cruise_speeds_for_sweeps(
        segments, swath_width=10.0, use_energy_optimal=True
    )
    wps = generate_smoothed_trajectory(
        segments,
        swath_width=10.0,
        use_energy_optimal=True,
        turn_resolution=5,
    )
    spray_velocities = [w.velocity for w in wps if w.is_spraying]
    assert max(spray_velocities) >= speeds[0] * 0.9
