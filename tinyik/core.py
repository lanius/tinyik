"""Core features."""

from numbers import Number

import autograd.numpy as np

from .component import Link, Joint
from .solver import FKSolver, IKSolver
from .optimizer import ScipyOptimizer


class Actuator(object):
    """Represents an actuator as a set of links and revolute joints."""

    def __init__(self, tokens, optimizer=None):
        """Create an actuator from specified link lengths and joint axes."""
        components = []
        for t in tokens:
            if isinstance(t, Number):
                components.append(Link([t, 0., 0.]))
            elif isinstance(t, list) or isinstance(t, np.ndarray):
                components.append(Link(t))
            elif isinstance(t, str) and t in {'x', 'y', 'z'}:
                components.append(Joint(t))
            else:
                raise ValueError(
                    'the arguments need to be '
                    'link length or joint axis: {}'.format(t)
                )

        self.fk = FKSolver(components)
        self.ik = IKSolver(
            self.fk, ScipyOptimizer() if optimizer is None else optimizer)

        self.angles = [0.] * len(
            [c for c in components if isinstance(c, Joint)]
        )
        self.components = components

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
        return self.fk.solve(self.angles)

    @ee.setter
    def ee(self, position):
        self.angles = self.ik.solve(self.angles, position)
