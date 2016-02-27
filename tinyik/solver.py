"""Solvers."""

from functools import reduce

import autograd.numpy as np

from .component import Joint


class FKSolver(object):
    """A forward kinematics solver."""

    def __init__(self, components):
        """Generate a FK solver from link and joint instances."""
        joint_indexes = [
            i for i, c in enumerate(components) if isinstance(c, Joint)
        ]

        def matrices(angles):
            joints = dict(zip(joint_indexes, angles))
            a = [joints.get(i, None) for i in range(len(components))]
            return [c.matrix(a[i]) for i, c in enumerate(components)]

        self._matrices = matrices

    def solve(self, angles):
        """Calculate a position of the end-effector and return it."""
        return reduce(
            lambda a, m: np.dot(m, a),
            reversed(self._matrices(angles)),
            np.array([0., 0., 0., 1.])
        )[:3]


class IKSolver(object):
    """An inverse kinematics solver."""

    def __init__(self, fk_solver, opt_cls, opt_params=None):
        """Generate an IK solver from a FK solver instance."""
        def distance_squared(angles, target):
            x = target - fk_solver.solve(angles)
            return np.sum(np.power(x, 2))

        if opt_params is None:
            opt_params = {}
        self.optimizer = opt_cls(distance_squared, **opt_params)

    def solve(self, angles0, target):
        """Calculate joint angles and returns it."""
        return self.optimizer.optimize(np.array(angles0), target)
