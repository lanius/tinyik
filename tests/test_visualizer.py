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


if __name__ == '__main__':
    visualize()
    visualize_with_target()
    large_visualize()
    large_visualize_with_target()
