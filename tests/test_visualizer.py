import numpy as np

import tinyik


tokens = [
    [.3, .0, .0], 'z', [.3, .0, .0], 'x', [.0, -.5, .0], 'x', [.0, -.5, .0]]


def visualize():
    leg = tinyik.Actuator(tokens)
    leg.angles = np.deg2rad([30, 45, -90])
    tinyik.visualize(leg)


def visualize_with_target():
    leg = tinyik.Actuator(tokens)
    leg.angles = np.deg2rad([30, 45, -90])
    tinyik.visualize(leg, target=[.8, .0, .8])


large_tokens = [
    [85., 80., 0.],
    'z',
    [500., 0., 0.],
    'z',
    [0., -500., 0.],
]


def large_visualize():
    arm = tinyik.Actuator(large_tokens)
    tinyik.visualize(arm, radius=15.)


def large_visualize_with_target():
    arm = tinyik.Actuator(large_tokens)
    tinyik.visualize(arm, target=[400., -300., 0.], radius=15.)


def visualize_with_z_axis():
    arm = tinyik.Actuator(
        ['z', [0, 0, 180.7], 'y', [-612.7, 0, 0], 'y', [-571.55, 0, 0],
        'y', [0, -174.15, 0], 'z', [0, 0, -119.85], 'y', [0, -116.55, 0]])
    tinyik.visualize(arm, radius=10.)


if __name__ == '__main__':
    visualize()
    visualize_with_target()
    large_visualize()
    large_visualize_with_target()
    visualize_with_z_axis()
