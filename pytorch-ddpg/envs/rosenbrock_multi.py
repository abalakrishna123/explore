"""Multidimensional version of Rosenbrock"""

import numpy as np
from numpy import exp, sqrt, cos, pi, sin

from base_function import BaseFunction

class Rosenbrock(BaseFunction):

    target_E = 0.
    target_coords = np.array([1., 1.])
    xmin = np.array([-10.,-10.])
    xmax = np.array([10.,10.])

    def getEnergy(self, coords):
        coords = self.coerceCoords(coords)

        return 100. * sum([(
            coords[:, i+1] - coords[:, i]**2)**2 + (coords[:, i]-1)**2
            for i in range(coords.shape[1] - 1)])

    def getEnergyGradient(self, coords):
        coords = self.coerceCoords(coords)

        E = self.getEnergy(coords)
        gradients = np.zeros(E.shape)
        for i in range(gradients.shape[1]):
            gradients[i] += self.getIthEnergyGradient(coords, i)
            gradients[i+1] += self.getIp1thEnergyGradient(coords, i)

        return E, gradients

    def coerceCoords(self, coords):
        if len(coords.shape) == 1:
            coords = coords.reshape(1, 2)
        assert len(coords.shape) <= 2, "Only allows 2d tensors (n x d). Found shape {}".format(coords.shape)
        return coords

    def getIthEnergyGradient(self, coords, i):
        return -400. * (coords[:, i+1] - coords[:, i]**2) * coords[:, i] + 2. * (coords[:, i]-1)

    def getIp1thEnergyGradient(self, coords, i):
        return 200. * (coords[:, i+1] - coords[:, i]**2)


if __name__ == "__main__":
    f = Rosenbrock()
    # f.test_potential(f.target_coords)
    print("")
    # f.test_potential(f.get_random_configuration())

    from base_function import makeplot2d
    v = 3.
    xmin = np.array([0.,0.])
    xmax = np.array([1.2,1.5])
    makeplot2d(f, xmin=-xmax, xmax=xmax, nx=30)
