"""Optimizers."""

import autograd.numpy as np
import autograd


class NewtonOptimizer(object):
    """An optimizer based on Newton's method."""

    def __init__(self, tol=1.48e-08, maxiter=50):
        """Generate an optimizer from an objective function."""
        self.tol = tol
        self.maxiter = maxiter

    def prepare(self, f):
        """Accept an objective function for optimization."""
        self.g = autograd.grad(f)
        self.h = autograd.hessian(f)

    def optimize(self, x0, target):
        """Calculate an optimum argument of an objective function."""
        x = x0
        for _ in range(self.maxiter):
            delta = np.linalg.solve(self.h(x, target), -self.g(x, target))
            x = x + delta
            if np.linalg.norm(delta) < self.tol:
                break
        return x


class SteepestDescentOptimizer(object):
    """An optimizer based on steepest descent method."""

    def __init__(self, tol=1.48e-08, maxiter=50, alpha=1):
        """Generate an optimizer from an objective function."""
        self.tol = tol
        self.maxiter = maxiter
        self.alpha = alpha

    def prepare(self, f):
        """Accept an objective function for optimization."""
        self.g = autograd.grad(f)

    def optimize(self, x0, target):
        """Calculate an optimum argument of an objective function."""
        x = x0
        for _ in range(self.maxiter):
            delta = self.alpha * self.g(x, target)
            x = x - delta
            if np.linalg.norm(delta) < self.tol:
                break
        return x


class ConjugateGradientOptimizer(object):
    """An optimizer based on conjugate gradient method."""

    def __init__(self, tol=1.48e-08, maxiter=50):
        """Generate an optimizer from an objective function."""
        self.tol = tol
        self.maxiter = maxiter

    def prepare(self, f):
        """Accept an objective function for optimization."""
        self.g = autograd.grad(f)
        self.h = autograd.hessian(f)

    def optimize(self, x0, target):
        """Calculate an optimum argument of an objective function."""
        x = x0
        for i in range(self.maxiter):
            g = self.g(x, target)
            h = self.h(x, target)
            if i == 0:
                alpha = 0
                m = g
            else:
                alpha = - np.dot(m, np.dot(h, g)) / np.dot(m, np.dot(h, m))
                m = g + np.dot(alpha, m)
            t = - np.dot(m, g) / np.dot(m, np.dot(h, m))
            delta = np.dot(t, m)
            x = x + delta
            if np.linalg.norm(delta) < self.tol:
                break
        return x
