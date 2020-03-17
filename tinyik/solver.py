"""Solvers."""

from functools import reduce
import sys

import autograd.numpy as np

from .component import Joint


class FKSolver:
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

    def __init__(self, fk_solver, optimizer):
        """Generate an IK solver from a FK solver instance."""
        def distance_squared(angles, target):
            x = target - fk_solver.solve(angles)
            return np.sum(np.power(x, 2))

        optimizer.prepare(distance_squared)
        self.optimizer = optimizer

    def solve(self, angles0, target):
        """Calculate joint angles and returns it."""
        return self.optimizer.optimize(np.array(angles0), target)


class CCDFKSolver(object):

    def __init__(self, components):
        joint_indexes = [
            i for i, c in enumerate(components) if isinstance(c, Joint)
        ]

        def matrices(angles):
            joints = dict(zip(joint_indexes, angles))
            a = [joints.get(i, None) for i in range(len(components))]
            return [c.matrix(a[i]) for i, c in enumerate(components)]

        self._matrices = matrices
        self.components = components
        self.joint_indexes = joint_indexes

    def solve(self, angles, p=None, index=None):
        if p is None:
            p = [0., 0., 0., 1.]
        if index is None:
            index = len(self.components) - 1
        return reduce(
            lambda a, m: np.dot(m, a),
            reversed(self._matrices(angles)[:index + 1]),
            np.array(p)
        )[:3]


class CCDIKSolver:

    def __init__(self, fk_solver, tol=1.48e-08, maxiter=50):
        self._fk_solver = fk_solver
        self.tol = tol
        self.maxiter = maxiter

    def solve(self, angles0, target):
        joint_indexes = list(reversed(self._fk_solver.joint_indexes))
        angles = angles0[:]
        pmap = {
            'x': [1., 0., 0., 0.],
            'y': [0., 1., 0., 0.],
            'z': [0., 0., 1., 0.]}
        prev_dist = sys.float_info.max
        for _ in range(self.maxiter):
            for i, idx in enumerate(joint_indexes):
                axis = self._fk_solver.solve(
                    angles,
                    p=pmap[self._fk_solver.components[idx].axis],
                    index=idx)
                pj = self._fk_solver.solve(angles, index=idx)
                ee = self._fk_solver.solve(angles)
                eorp = self.p_on_rot_plane(ee, pj, axis)
                torp = self.p_on_rot_plane(target, pj, axis)
                ve = (eorp - pj) / np.linalg.norm(eorp - pj)
                vt = (torp - pj) / np.linalg.norm(torp - pj)
                a = np.arccos(np.dot(vt, ve))
                sign = 1 if np.dot(axis, np.cross(ve, vt)) > 0 else -1
                angles[-(i + 1)] += (a * sign)

            ee = self._fk_solver.solve(angles)
            dist = np.linalg.norm(target - ee)
            delta = prev_dist - dist
            if delta < self.tol:
                break
            prev_dist = dist

        return angles

    def p_on_rot_plane(self, p, joint_pos, joint_axis):
        ua = joint_axis / np.linalg.norm(joint_axis)
        return p - (np.dot(p - joint_pos, ua) * ua)
