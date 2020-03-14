from tinyik import Link, Joint, FKSolver, CCDFKSolver, CCDIKSolver

from .utils import x, y, z, theta, approx_eq


components = [Joint('z'), Link([1., 0., 0.]), Joint('y'), Link([1., 0., 0.])]
predicted = [2., 0., 0.]


def test_fk():
    fk = FKSolver(components)
    assert all(fk.solve([0., 0.]) == predicted)

    assert approx_eq(fk.solve([theta, theta]), [x, y, -z])
    assert approx_eq(fk.solve([-theta, -theta]), [x, -y, z])


def test_ccd_fk():
    fk = CCDFKSolver(components)
    assert all(fk.solve([0., 0.]) == predicted)

    assert approx_eq(fk.solve([theta, theta]), [x, y, -z])
    assert approx_eq(fk.solve([-theta, -theta]), [x, -y, z])


def test_ccd_ik():
    fk = CCDFKSolver(components)
    ik = CCDIKSolver(fk)
    assert approx_eq(ik.solve([0., 0.], [x, y, -z]), [theta, theta])
    assert approx_eq(ik.solve([0., 0.], [x, -y, z]), [-theta, -theta])
