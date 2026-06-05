"""Unit tests for Phase I ULog grading helpers (no .ulg required)."""

import math

import numpy as np

from analysis.grade_phase_1 import (
    PWM_DEFAULT_FAIL,
    CheckResult,
    Phase1GradeReport,
    quaternion_to_yaw_rad,
)
from trajectory.drift_model import crab_angle_rad, vehicle_yaw_deg_from_track


def test_quaternion_yaw_identity():
    # Identity quaternion → yaw 0
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert abs(quaternion_to_yaw_rad(q)) < 1e-6


def test_quaternion_yaw_90_deg_north():
    # 90° yaw about down (NED): q = [cos(45°), 0, 0, sin(45°)]
    ang = math.radians(90)
    q = np.array([math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)])
    yaw = quaternion_to_yaw_rad(q)
    assert abs(math.degrees(yaw) - 90.0) < 2.0


def test_expected_crab_matches_vehicle_yaw_helper():
    heading = 0.0
    crab = crab_angle_rad(heading, 5.0, 180.0, 8.0)
    yaw = vehicle_yaw_deg_from_track(heading, crab)
    assert 0.0 <= yaw < 360.0


def test_phase1_report_all_passed():
    r = Phase1GradeReport(
        ulog_path='test.ulg',
        checks=[
            CheckResult('xte', True, 'ok'),
            CheckResult('crab', True, 'ok'),
        ],
    )
    assert r.passed


def test_pwm_threshold_constant():
    assert PWM_DEFAULT_FAIL < 2000
