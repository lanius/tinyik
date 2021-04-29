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
    'y',
    [500., 0., 0.],
    'y',
    [0., -500., 0.],
    'y'
]


def large_visualize():
    arm = tinyik.Actuator(large_tokens)
    tinyik.visualize(arm, radius=15.)


if __name__ == '__main__':
    visualize()
    visualize_with_target()
    large_visualize()
