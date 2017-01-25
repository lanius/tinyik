import numpy as np

from tinyik import (
    Link, Joint,
    FKSolver, IKSolver,
    NewtonOptimizer,
    SteepestDescentOptimizer,
    ConjugateGradientOptimizer,
    ScipyOptimizer, ScipySmoothOptimizer
)

from .utils import x, y, z, theta, approx_eq


def build_ik_solver(optimizer_instance):
    fk = FKSolver([
        Joint('z'), Link([1., 0., 0.]), Joint('y'), Link([1., 0., 0.])
    ])
    return IKSolver(fk, optimizer_instance)


def test_inverse_kinematics_with_newton():
    ik = build_ik_solver(NewtonOptimizer())
    assert approx_eq(ik.solve([theta, theta], [2., 0., 0.]), [0., 0.])
    assert approx_eq(ik.solve([0., 0.], [x, y, -z]), [theta, theta])
    assert approx_eq(ik.solve([0., 0.], [x, -y, z]), [-theta, -theta])


def test_inverse_kinematics_with_steepest_descent():
    ik = build_ik_solver(SteepestDescentOptimizer(maxiter=100, alpha=0.1))
    assert approx_eq(ik.solve([theta, theta], [2., 0., 0.]), [0., 0.])
    assert approx_eq(ik.solve([0., 0.], [x, y, -z]), [theta, theta])
    assert approx_eq(ik.solve([0., 0.], [x, -y, z]), [-theta, -theta])


def test_inverse_kinematics_with_conjugate_gradient():
    ik = build_ik_solver(ConjugateGradientOptimizer())
    assert approx_eq(ik.solve([theta, theta], [2., 0., 0.]), [0., 0.])
    assert approx_eq(ik.solve([0., 0.], [x, y, -z]), [theta, theta])
    assert approx_eq(ik.solve([0., 0.], [x, -y, z]), [-theta, -theta])


def test_inverse_kinematics_with_scipy():
    ik = build_ik_solver(ScipyOptimizer())
    assert approx_eq(ik.solve([theta, theta], [2., 0., 0.]), [0., 0.])
    assert approx_eq(ik.solve([0., 0.], [x, y, -z]), [theta, theta])
    assert approx_eq(ik.solve([0., 0.], [x, -y, z]), [-theta, -theta])


def test_inverse_kinematics_with_scipy_smooth():
    ik = build_ik_solver(ScipySmoothOptimizer(smooth_factor=0.))
    assert approx_eq(ik.solve([theta, theta], [2., 0., 0.]), [0., 0.])
    assert approx_eq(ik.solve([0., 0.], [x, y, -z]), [theta, theta])
    assert approx_eq(ik.solve([0., 0.], [x, -y, z]), [-theta, -theta])

    def approx_roughly_eq(a, b):
        return all(np.round(a, 1) == np.round(b, 1))

    ik = build_ik_solver(ScipySmoothOptimizer())
    assert approx_roughly_eq(ik.solve([theta, theta], [2., 0., 0.]), [0., 0.])
    assert approx_roughly_eq(ik.solve([0., 0.], [x, y, -z]), [theta, theta])
    assert approx_roughly_eq(ik.solve([0., 0.], [x, -y, z]), [-theta, -theta])
