"""Probe initial conditions must implement the predeclared world azimuth."""
import importlib.util
from pathlib import Path
import math
import numpy as np
import pytest
from adaptive_drone_patrol.payload_contract import rotation

@pytest.mark.parametrize('angle', [0, 5, 10, 45, 89, 90, 91, 120, 170])
@pytest.mark.parametrize('direction', [0., math.pi/2, 2.33])
def test_initial_suspension_quaternion_matches_declared_direction(angle, direction):
    path = Path(__file__).parents[1]/'scripts/probe_payload_ball.py'
    spec = importlib.util.spec_from_file_location('ball_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q = module.initial_suspension_quaternion(angle, direction)
    actual = rotation(q) @ [0, 0, -1]
    a = math.radians(angle)
    expected = [math.sin(a)*math.cos(direction), math.sin(a)*math.sin(direction), -math.cos(a)]
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)


def test_speed_extrema_use_each_body_or_rotor_vector_not_matrix_norm():
    path = Path(__file__).parents[1]/'scripts/probe_payload_ball.py'
    spec = importlib.util.spec_from_file_location('ball_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row=dict(drone_velocity=[0,0,15],package_velocity=[0,0,14],rod_velocity=[0,0,13],
             rotor_com_velocity=[[0,0,15]]*4,drone_angular_velocity=[0,0,150],
             package_angular_velocity=[0,0,149],rotor_angular_velocity=[[0,0,150]]*4,
             dofs_velocity=[1000]*12)
    assert module.physical_speed_extrema(row)==(15.,150.)
    row['package_angular_velocity']=[0,0,201]
    assert module.physical_speed_extrema(row)==(15.,201.)
