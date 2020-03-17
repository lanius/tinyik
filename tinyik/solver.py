"""Solvers."""

from functools import reduce
import sys

import autograd.numpy as np

from .component import ComponentList


class FKSolver:
    """A forward kinematics solver."""

    def __init__(self, components):
        """Generate a FK solver from link and joint instances."""
        self.components = ComponentList(components)

    def matrices(self, angles):
        map_ = dict(zip(self.components.joint_indexes, angles))
        return [
            c.matrix(map_.get(i, None)) for i, c in enumerate(self.components)]

    def solve(self, angles, p=None, index=None):
        """Calculate a position of the end-effector and return it."""
        if p is None:
            p = [0., 0., 0., 1.]
        if index is None:
            index = len(self.components) - 1
        return reduce(
            lambda a, m: np.dot(m, a),
            reversed(self.matrices(angles)[:index + 1]),
            np.array(p)
        )[:3]


class CCDIKSolver:

    def __init__(self, fk_solver, tol=1.48e-08, maxiter=50):
        self.fk_solver = fk_solver
        self.tol = tol
        self.maxiter = maxiter

    def solve(self, angles0, target):
        joint_indexes = list(reversed(self.fk_solver.components.joint_indexes))
        angles = angles0[:]
        prev_dist = sys.float_info.max
        for _ in range(self.maxiter):
            for i, idx in enumerate(joint_indexes):
                a = self.fk_solver.solve(
                        angles,
                        p=np.append(
                            self.fk_solver.components[idx].axis[:], 0.),
                        index=idx)
                axis = a / np.linalg.norm(a)
                joint_p = self.fk_solver.solve(angles, index=idx)
                ee = self.fk_solver.solve(angles)
                eorp = self.p_on_rot_plane(ee, joint_p, axis)
                torp = self.p_on_rot_plane(target, joint_p, axis)
                ve = (eorp - joint_p) / np.linalg.norm(eorp - joint_p)
                vt = (torp - joint_p) / np.linalg.norm(torp - joint_p)
                angle = np.arccos(np.dot(vt, ve))
                sign = 1 if np.dot(axis, np.cross(ve, vt)) > 0 else -1
                angles[-(i + 1)] += (angle * sign)

            ee = self.fk_solver.solve(angles)
            dist = np.linalg.norm(target - ee)
            delta = prev_dist - dist
            if delta < self.tol:
                break
            prev_dist = dist

        return angles

    def p_on_rot_plane(self, p, joint_p, joint_axis):
        return p - (np.dot(p - joint_p, joint_axis) * joint_axis)


class OptimizationBasedIKSolver:
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
