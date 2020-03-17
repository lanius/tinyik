"""Core features."""

from numbers import Number

import numpy as np

from .component import Link, Joint, ComponentList
from .solver import FKSolver, CCDIKSolver


def create(tokens):
    components = []
    for t in tokens:
        if isinstance(t, Number):
            components.append(Link([t, 0., 0.]))
        elif isinstance(t, list) or isinstance(t, np.ndarray):
            components.append(Link(t))
        elif t in {'x', 'y', 'z'}:
            components.append(Joint(t))
        else:
            raise ValueError(
                'the arguments need to be '
                'link length or joint axis: {}'.format(t)
            )

    fk_solver = FKSolver(components)
    ik_solver = CCDIKSolver(fk_solver)
    return Actuator(components, fk_solver, ik_solver)


class Actuator:
    """Represents an actuator as a set of links and revolute joints."""

    def __init__(self, components, fk_solver, ik_solver):
        self.components = ComponentList(components)
        self.fk = fk_solver
        self.ik = ik_solver

    @property
    def angles(self):
        """The joint angles."""
        return [j.angle for j in self.components.joints]

    @angles.setter
    def angles(self, angles):
        for j, a in zip(self.components.joints, angles):
            j.angle = a

    @property
    def ee(self):
        """The end-effector position."""
        return self.fk.solve(self.angles)

    @ee.setter
    def ee(self, position):
        self.angles = self.ik.solve(self.angles, position)
