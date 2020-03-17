"""Components for an actuator."""

try:
    import autograd.numpy as np
except ImportError:
    import numpy as np


class Link:
    """Represents a link."""

    def __init__(self, coord):
        """Create a link from a specified coordinate."""
        self.coord = coord

    def matrix(self, _):
        """Return translation matrix in homogeneous coordinates."""
        x, y, z = self.coord
        return np.array([
            [1., 0., 0., x],
            [0., 1., 0., y],
            [0., 0., 1., z],
            [0., 0., 0., 1.]
        ])


class Joint:
    """Represents a revolute joint."""

    def __init__(self, axis):
        """Create a revolute joint from a specified axis."""
        if isinstance(axis, str) and axis in {'x', 'y', 'z', '-x', '-y', '-z'}:
            self.axis = {
                'x': [1., 0., 0.],
                'y': [0., 1., 0.],
                'z': [0., 0., 1.],
                '-x': [-1., 0., 0.],
                '-y': [0., -1., 0.],
                '-z': [0., 0., -1.]
                }[axis]
        else:
            axis_norm = np.linalg.norm(axis)
            self.axis = axis if axis_norm == 1 else (axis / axis_norm)
        self.angle = 0.

    def matrix(self, angle):
        """Return rotation matrix in homogeneous coordinates."""
        return self._rot(self.axis, angle)

    def _rot(self, axis, angle):
        x, y, z, = axis
        return np.array([  # Rodrigues
            [
                np.cos(angle) + (x**2 * (1 - np.cos(angle))),
                (x * y * (1 - np.cos(angle))) - (z * np.sin(angle)),
                (x * z * (1 - np.cos(angle))) + (y * np.sin(angle)),
                0.
            ], [
                (y * x * (1 - np.cos(angle))) + (z * np.sin(angle)),
                np.cos(angle) + (y**2 * (1 - np.cos(angle))),
                (y * z * (1 - np.cos(angle))) - (x * np.sin(angle)),
                0.
            ], [
                (z * x * (1 - np.cos(angle))) - (y * np.sin(angle)),
                (z * y * (1 - np.cos(angle))) + (x * np.sin(angle)),
                np.cos(angle) + (z**2 * (1 - np.cos(angle))),
                0.
            ], [0., 0., 0., 1.]
        ])


class RestrictedJoint(Joint):

    def __init__(self, axis, lower_limit, upper_limit):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        super().__init__(axis)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        self._angle = np.clip(angle, self.lower_limit, self.upper_limit)


class ComponentList(list):

    @property
    def joints(self):
        return [c for c in self if isinstance(c, Joint)]

    @property
    def joint_indexes(self):
        return [self.index(j) for j in self.joints]
