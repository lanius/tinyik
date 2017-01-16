"""Optimizers."""

import autograd.numpy as np
import autograd

import scipy.optimize

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

class Scipy_Minimize(object):
    """An optimizer based on scipy.optimize.minimize."""

    def __init__(self, tol=1.48e-08, maxiter=50, method='BFGS'):
        """Generate an optimizer from an objective function."""
        self.tol = tol
        self.maxiter = maxiter
        self.method = method

    def prepare(self, f):
        """Accept an objective function for optimization."""
        self.f = f

    def optimize(self, angles0, target):
        """Calculate an optimum argument of an objective function."""
        def new_objective(angles, target=target):
            return self.f(angles, target)

        return scipy.optimize.minimize(new_objective,angles0,
                                       method=self.method,
                                       tol=self.tol,
                                       options={'maxiter':self.maxiter}).x

class Scipy_Minimize_Smooth(object):
    """An optimizer based on scipy.optimize.minimize."""

    def __init__(self, tol=1.48e-08, maxiter=50, bounds=None,
                 smooth_factor=0.1, method='L-BFGS-B'):
        """Generate an optimizer from an objective function."""
        self.tol = tol
        self.maxiter = maxiter
        self.bounds = bounds
        self.smooth_factor = smooth_factor
        self.method = method

    def prepare(self, f):
        """Accept an objective function for optimization."""
        self.f = f

    def optimize(self, angles0, target):
        """Calculate an optimum argument of an objective function."""
        def new_objective(angles, target=target):
            a = angles - angles0
            return self.f(angles, target)+self.smooth_factor*np.sum(np.power(a, 2))

        # return scipy.optimize.minimize(new_objective,angles0,method='BFGS',options={'gtol':1.48e-08}).x
        return scipy.optimize.minimize(new_objective,angles0,
                                       method=self.method,
                                       bounds=self.bounds,
                                       tol=self.tol,
                                       options={'maxiter':self.maxiter}).x
        
