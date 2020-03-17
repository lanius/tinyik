import numpy as np

from tinyik import create

from .utils import x, y, z, theta, approx_eq


def test_actuator_instantiation():
    two_joints_arm = create(['z', 1., 'z', 1.])
    assert len(two_joints_arm.angles) == 2
    assert np.all(two_joints_arm.angles == [0., 0.])
    assert np.all(two_joints_arm.ee == [2., 0., 0.])

    three_joints_arm = create(['x', 1., 'y', 1., 'z', 1.])
    assert len(three_joints_arm.angles) == 3
    assert np.all(three_joints_arm.angles == [0., 0., 0.])
    assert np.all(three_joints_arm.ee == [3., 0., 0.])

    y_axis_dir_arm = create(['z', [0., 1., 0.], 'z', [0., 1., 0.]])
    assert np.all(y_axis_dir_arm.ee == [0., 2., 0.])


def test_actuator_angles():
    arm = create(['z', 1., 'y', 1.])
    arm.angles = [theta, theta]
    assert approx_eq(arm.angles, [theta, theta])
    assert approx_eq(arm.ee, [x, y, -z])


def test_actuator_ee():
    arm = create(['z', 1., 'y', 1.])
    arm.ee = [x, -y, z]
    assert approx_eq(arm.ee, [x, -y, z])
    assert approx_eq(arm.angles, [-theta, -theta])
