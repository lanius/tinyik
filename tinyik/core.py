"""Core features."""

from numbers import Number
from functools import reduce

import autograd.numpy as np
import autograd


class Actuator(object):
    """Represents an actuator as a set of links and revolute joints."""

    def __init__(self, *args):
        """Create an actuator from specified link lengths and joint axes."""
        components = []
        for arg in args:
            if isinstance(arg, Number):
                components.append(Link(arg))
            elif isinstance(arg, str) and arg in {'x', 'y', 'z'}:
                components.append(Joint(arg))
            else:
                raise ValueError(
                    'the arguments need to be '
                    'link length or joint axis: {}'.format(arg)
                )

        self._fk = FKSolver(components)
        self._ik = IKSolver(self._fk)

        self.angles = [0.] * len(
            [c for c in components if isinstance(c, Joint)]
        )

    @property
    def angles(self):
        """The joint angles."""
        return self._angles

    @angles.setter
    def angles(self, angles):
        self._angles = np.array(angles)

    @property
    def ee(self):
        """The end-effector position."""
        return self._fk.solve(self.angles)

    @ee.setter
    def ee(self, position):
        self.angles = self._ik.solve(position, self.angles)


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

    def __init__(self, fk_solver):
        """Generate an IK solver from a FK solver instance."""
        def distance_squared(target, angles):
            x = target - fk_solver.solve(angles)
            return np.sum(np.power(x, 2))

        self._g = autograd.grad(distance_squared, argnum=1)
        self._h = autograd.hessian(distance_squared, argnum=1)

    def solve(self, target, angles0, tol=1.48e-08, maxiter=50):
        """Calculate joint angles and returns it."""
        angles = np.array(angles0)
        for _ in range(maxiter):
            delta = np.linalg.solve(
                self._h(target, angles), -self._g(target, angles))
            angles = angles + delta
            if np.linalg.norm(delta) < tol:
                break
        return angles


class Link(object):
    """Represents a link."""

    def __init__(self, length):
        """Create a link from a specified length."""
        self.length = length

    def matrix(self, _):
        """Return translation matrix in homogeneous coordinates."""
        return np.array([
            [1., 0., 0., self.length],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])


class Joint(object):
    """Represents a revolute joint."""

    def __init__(self, axis):
        """Create a revolute joint from a specified axis."""
        self.axis = axis

    def matrix(self, angle):
        """Return rotation matrix in homogeneous coordinates."""
        _rot_mat = {
            'x': self._x_rot,
            'y': self._y_rot,
            'z': self._z_rot
        }
        return _rot_mat[self.axis](angle)

    def _x_rot(self, angle):
        return np.array([
            [1., 0., 0., 0.],
            [0., np.cos(angle), -np.sin(angle), 0.],
            [0., np.sin(angle), np.cos(angle), 0.],
            [0., 0., 0., 1.]
        ])

    def _y_rot(self, angle):
        return np.array([
            [np.cos(angle), 0., np.sin(angle), 0.],
            [0., 1., 0., 0.],
            [-np.sin(angle), 0., np.cos(angle), 0.],
            [0., 0., 0., 1.]
        ])

    def _z_rot(self, angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), 0., 0.],
            [np.sin(angle), np.cos(angle), 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])
