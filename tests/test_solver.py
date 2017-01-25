from tinyik import Link, Joint, FKSolver

from .utils import x, y, z, theta, approx_eq


def test_forward_kinematics():
    fk = FKSolver([
        Joint('z'), Link([1., 0., 0.]), Joint('y'), Link([1., 0., 0.])
    ])
    assert all(fk.solve([0., 0.]) == [2., 0., 0.])

    assert approx_eq(fk.solve([theta, theta]), [x, y, -z])
    assert approx_eq(fk.solve([-theta, -theta]), [x, -y, z])
