import numpy as np


x = (3. + (2. * np.sqrt(3.))) / 4.
y = (2. + np.sqrt(3.)) / 4.
z = .5

theta = np.pi / 6


def approx_eq(a, b):
    return all(np.round(a, 5) == np.round(b, 5))
