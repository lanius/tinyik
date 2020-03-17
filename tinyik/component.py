"""Components for an actuator."""

import autograd.numpy as np


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
