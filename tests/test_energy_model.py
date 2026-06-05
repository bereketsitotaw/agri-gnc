import math

from shapely.geometry import LineString

from trajectory.energy_model import (
    AircraftEnergyParams,
    energy_per_meter,
    instantaneous_power_w,
    mass_after_spray_pass,
    optimal_cruise_speed,
    sweep_mass_schedule,
    total_mass_kg,
)


def test_heavy_optimal_speed_higher_than_light():
    params = AircraftEnergyParams()
    v_heavy = optimal_cruise_speed(30.0, params)
    v_light = optimal_cruise_speed(15.0, params)
    assert v_heavy > v_light


def test_energy_per_meter_has_minimum_in_speed_range():
    params = AircraftEnergyParams()
    m = 25.0
    speeds = [3.0, 5.0, 7.0, 9.0, 11.0]
    costs = [energy_per_meter(m, v, params) for v in speeds]
    assert min(costs) < costs[0]
    assert min(costs) < costs[-1]


def test_mass_decreases_after_spray_pass():
    params = AircraftEnergyParams(dry_mass_kg=15.0, payload_mass_kg=15.0)
    m0 = total_mass_kg(1.0, params)
    m1 = mass_after_spray_pass(m0, sweep_length_m=100.0, swath_width_m=10.0,
                               application_rate_l_per_m2=0.02, params=params)
    assert m1 < m0
    assert m1 >= params.dry_mass_kg


def test_sweep_mass_schedule_monotonic_decrease():
    segments = [
        LineString([(0, 0), (100, 0)]),
        LineString([(100, 10), (0, 10)]),
        LineString([(0, 20), (100, 20)]),
    ]
    # 0.02 L/m² on 100×10 m passes would empty a 15 L tank in one sweep; use a lighter rate
    masses = sweep_mass_schedule(segments, swath_width_m=10.0, application_rate_l_per_m2=0.005)
    assert len(masses) == 3
    assert masses[0] > masses[1] > masses[2]


def test_instantaneous_power_increases_with_mass_at_fixed_speed():
    params = AircraftEnergyParams()
    p_light = instantaneous_power_w(15.0, 8.0, params)
    p_heavy = instantaneous_power_w(30.0, 8.0, params)
    assert p_heavy > p_light
