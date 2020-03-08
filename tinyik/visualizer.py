import numpy as np

try:
    import open3d as o3d  # the extra feature
except ImportError:
    pass


def translate(p):
    x, y, z = p
    return np.array([
            [1., 0., 0., x],
            [0., 1., 0., y],
            [0., 0., 1., z],
            [0., 0., 0., 1.]
        ])


def rotate(axis, angle):
    x, y, z = axis
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


def create_sphere(p, r=.1, color=None):
    if color is None:
        color = [.8, .8, .8]
    geo = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    geo.compute_vertex_normals()
    geo.paint_uniform_color(color)
    geo.transform(translate(p))
    return geo


class GeoComponent:

    child = None

    def tip(self, link_color=None):
        return create_sphere(
            [0., 0., 0.],
            color=[.8, .8, .8] if link_color is None else link_color)

    def geo(self, mat=None, link_color=None):
        geo = self.base_geo(link_color)
        if mat is not None:
            geo.transform(mat)
            mat = mat @ self.mat()
        else:
            mat = self.mat()
        if self.child is None:
            return [geo] + [self.tip(link_color).transform(mat)]
        else:
            return [geo] + self.child.geo(mat, link_color)


class Link(GeoComponent):

    def __init__(self, c):
        self.c = c

    def base_geo(self, link_color=None):
        norm = np.linalg.norm(self.c.coord)
        base = np.array([0., 0., norm])
        cross = np.cross(base, self.c.coord)
        axis = cross / np.linalg.norm(cross)
        angle = np.arccos(np.dot(base, self.c.coord) / (norm ** 2))
        geo = o3d.geometry.TriangleMesh.create_cylinder(
            radius=.05, height=norm)
        geo.compute_vertex_normals()
        geo.paint_uniform_color(
            [.8, .8, .8] if link_color is None else link_color)
        geo.transform(rotate(axis, angle) @ translate(base / 2))
        return geo

    def mat(self):
        return self.c.matrix(None)


class Joint(GeoComponent):

    def __init__(self, c):
        self.c = c
        self.angle = 0.

    def base_geo(self, _=None):
        geo = o3d.geometry.TriangleMesh.create_cylinder(radius=.1, height=.2)
        geo.compute_vertex_normals()
        geo.paint_uniform_color([.2, .2, .9])
        rx = {
            'x': [0., 1., 0.],
            'y': [1., 0., 0.],
            'z': [0., 0., 1.],
        }
        geo.transform(rotate(rx[self.c.axis], np.pi / 2))
        return geo

    def mat(self):
        return self.c.matrix(self.angle)


def visualize(actuator, target=None):
    root = None
    p = None
    joints = []
    for c in actuator.components:
        if hasattr(c, 'axis'):
            gc = Joint(c)
            joints.append(gc)
        else:
            gc = Link(c)

        if root is None:
            root = gc
            p = gc
        else:
            p.child = gc
            p = gc

    for j, a in zip(joints, actuator.angles):
        j.angle = a

    if target:
        geos = root.geo(link_color=[.5, .5, .5])
        actuator.ee = target
        for j, a in zip(joints, actuator.angles):
            j.angle = a
        geos += root.geo()
        geos += [create_sphere(target, r=.12, color=[.8, .2, .2])]
    else:
        geos = root.geo()

    o3d.visualization.draw_geometries(
        geos, window_name='tinyik vizualizer', width=640, height=480)
