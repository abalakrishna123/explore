import numpy as np
from numpy import exp, sqrt, cos, pi, sin

from .base_function import BaseFunction

class Rosenbrock(BaseFunction):
    target_E = 0.
    target_coords = np.array([1., 1.])
    xmin = np.array([-10.,-10.])
    xmax = np.array([10.,10.])

    def getEnergy(self, coords, theta):
        x, y = coords
        w, b = theta
        return w * (y - x**2)**2 + (x-b)**2

    def getEnergyGradient(self, coords, theta):
        E = self.getEnergy(coords, theta)

        x, y = coords
        w, b = theta

        dx = -4. * w * (y - x**2) * x + 2. * (x-b)
        dy = 2. * w * (y - x**2)
        return E, np.array([dx, dy])



if __name__ == "__main__":
    f = Rosenbrock()
    # f.test_potential(f.target_coords)
    # print ""
    # f.test_potential(f.get_random_configuration())

    from base_function import makeplot2d
    v = 3.
    xmin = np.array([0.,0.])
    xmax = np.array([1.2,1.5])
    makeplot2d(f, xmin=-xmax, xmax=xmax, nx=30)
