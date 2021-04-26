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


if __name__ == '__main__':
    visualize()
    visualize_with_target()
