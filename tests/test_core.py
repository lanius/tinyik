from tinyik import Actuator

from .utils import x, y, z, theta, approx_eq


def test_actuator_instantiation():
    two_joints_arm = Actuator(['z', 1., 'z', 1.])
    assert len(two_joints_arm.angles) == 2
    assert all(two_joints_arm.angles == 0.)
    assert all(two_joints_arm.ee == [2., 0., 0.])

    three_joints_arm = Actuator(['x', 1., 'y', 1., 'z', 1.])
    assert len(three_joints_arm.angles) == 3
    assert all(three_joints_arm.angles == 0.)
    assert all(three_joints_arm.ee == [3., 0., 0.])

    y_axis_dir_arm = Actuator(['z', [0., 1., 0.], 'z', [0., 1., 0.]])
    assert all(y_axis_dir_arm.ee == [0., 2., 0.])


def test_actuator_angles():
    arm = Actuator(['z', 1., 'y', 1.])
    arm.angles = [theta, theta]
    assert approx_eq(arm.angles, [theta, theta])
    assert approx_eq(arm.ee, [x, y, -z])


def test_actuator_ee():
    arm = Actuator(['z', 1., 'y', 1.])
    arm.ee = [x, -y, z]
    assert approx_eq(arm.ee, [x, -y, z])
    assert approx_eq(arm.angles, [-theta, -theta])


def test_multiple_actuator_ee():
    first_arm = Actuator(
        ['y', [0., 0., 0.01], 'x', [0., 0., 1.11], 'x', [0., 0., 1.225]])
    second_arm = Actuator(
        ['x', [0., 0., 1.175], 'x', [0., 0., 1.225], 'x', [0., 0., .1]])
    angles = [0, -1, 2.5]
    first_arm.angles = angles[:]
    first_arm.ee = first_arm.ee + [0., 0., 0.]
    assert approx_eq(first_arm.angles, angles)
    first_arm.ee = first_arm.ee + [0., 0., 0.]
    assert approx_eq(first_arm.angles, angles)
    assert approx_eq(second_arm.angles, [0., 0., 0.])
