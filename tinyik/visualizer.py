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


def create_sphere(p, radius, color=None):
    if color is None:
        color = [.8, .8, .8]
    geo = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    geo.compute_vertex_normals()
    geo.paint_uniform_color(color)
    geo.transform(translate(p))
    return geo


class GeoComponent:

    child = None
    radius = .1

    def tip(self, link_color=None):
        return create_sphere(
            [0., 0., 0.],
            radius=self.radius*2,
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

    def __init__(self, c, radius):
        self.c = c
        self.radius = radius

    def base_geo(self, link_color=None):
        norm = np.linalg.norm(self.c.coord)

        geo = o3d.geometry.TriangleMesh.create_cylinder(
            radius=self.radius, height=norm)
        geo.compute_vertex_normals()
        geo.paint_uniform_color(
            [.8, .8, .8] if link_color is None else link_color)

        # Calculate transformation matrix for cylinder
        # With help from https://stackoverflow.com/a/59829173
        def get_cross_prod_mat(vector):
            return np.array([
                [0, -vector[2], vector[1]], 
                [vector[2], 0, -vector[0]],
                [-vector[1], vector[0], 0],
            ])
        cylinder_dir_unit_vector = self.c.coord / norm

        # Unit vector for "up" direction
        z_unit_vector = np.array([0,0,1])
        z_rotation_mat = get_cross_prod_mat(z_unit_vector)

        z_c_vec = np.matmul(z_rotation_mat, cylinder_dir_unit_vector)
        z_c_vec_mat = get_cross_prod_mat(z_c_vec)

        # Added np.abs to ensure that unit vector that is aligned with any axis in the negative direction does not 
        rotation_mat = np.eye(3,3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat)/((1 + np.abs(np.dot(z_unit_vector, cylinder_dir_unit_vector))))
        
        cylinder_transform_mat = np.vstack((np.hstack((rotation_mat, np.transpose(np.array([self.c.coord])/2))), np.array([0,0,0,1])))

        geo.transform(cylinder_transform_mat)
        return geo

    def mat(self):
        return self.c.matrix(None)


class Joint(GeoComponent):

    def __init__(self, c, radius):
        self.c = c
        self.radius = radius
        self.angle = 0.

    def base_geo(self, _=None):
        geo = o3d.geometry.TriangleMesh.create_cylinder(
            radius=self.radius*2, height=self.radius*4)
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


def build_geos(actuator, target=None, radius=.05):
    root = None
    p = None
    joints = []
    for c in actuator.components:
        if hasattr(c, 'axis'):
            gc = Joint(c, radius)
            joints.append(gc)
        else:
            gc = Link(c, radius)

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
        geos += [create_sphere(target, radius=radius*2.4, color=[.8, .2, .2])]
    else:
        geos = root.geo()

    return geos


def visualize(actuator, target=None, radius=.05):
    geos = build_geos(actuator, target, radius)
    o3d.visualization.draw_geometries(
        geos, window_name='tinyik vizualizer', width=640, height=480)
