import numpy as np

from tinyik.core import (
    Actuator, FKSolver, IKSolver,
    SDOptimizer,
    Link, Joint
)


x = (3. + (2. * np.sqrt(3.))) / 4.
y = (2. + np.sqrt(3.)) / 4.
z = .5

theta = np.pi / 6


def test_actuator_instantiation():
    two_joints_arm = Actuator('z', 1., 'z', 1.)
    assert len(two_joints_arm.angles) == 2
    assert all(two_joints_arm.angles == 0.)
    assert all(two_joints_arm.ee == [2., 0., 0.])

    three_joints_arm = Actuator('x', 1., 'y', 1., 'z', 1.)
    assert len(three_joints_arm.angles) == 3
    assert all(three_joints_arm.angles == 0.)
    assert all(three_joints_arm.ee == [3., 0., 0.])


def test_actuator_angles():
    arm = Actuator('z', 1., 'y', 1.)
    arm.angles = [theta, theta]
    assert approx_eq(arm.angles, [theta, theta])
    assert approx_eq(arm.ee, [x, y, -z])


def test_actuator_ee():
    arm = Actuator('z', 1., 'y', 1.)
    arm.ee = [x, -y, z]
    assert approx_eq(arm.ee, [x, -y, z])
    assert approx_eq(arm.angles, [-theta, -theta])


def test_forward_kinematics():
    fk = FKSolver([Joint('z'), Link(1.), Joint('y'), Link(1.)])
    assert all(fk.solve([0., 0.]) == [2., 0., 0.])

    assert approx_eq(fk.solve([theta, theta]), [x, y, -z])
    assert approx_eq(fk.solve([-theta, -theta]), [x, -y, z])


def test_inverse_kinematics():
    fk = FKSolver([Joint('z'), Link(1.), Joint('y'), Link(1.)])
    ik = IKSolver(fk)
    assert approx_eq(ik.solve([theta, theta], [2., 0., 0.]), [0., 0.])

    assert approx_eq(ik.solve([0., 0.], [x, y, -z]), [theta, theta])
    assert approx_eq(ik.solve([0., 0.], [x, -y, z]), [-theta, -theta])


def test_sd_optimizer():
    fk = FKSolver([Joint('z'), Link(1.), Joint('y'), Link(1.)])
    ik = IKSolver(fk, SDOptimizer, {'maxiter': 100, 'alpha': 0.1})
    assert approx_eq(ik.solve([theta, theta], [2., 0., 0.]), [0., 0.])

    assert approx_eq(ik.solve([0., 0.], [x, y, -z]), [theta, theta])
    assert approx_eq(ik.solve([0., 0.], [x, -y, z]), [-theta, -theta])


def approx_eq(a, b):
    return all(np.round(a, 5) == np.round(b, 5))
