import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Solver:

    def __init__(self, mesh, values, boundary_values):
        """
        Constructor for Solver
        :param mesh: numpy.meshgrid
        Mesh grid
        :param values: 2D array
        Initial values on grid
        :param boundary_values: 2D array
        Boundary values on grid, else zero
        """
        self.mesh = mesh
        self.values = values
        self.boundary_values = boundary_values

    @staticmethod
    def make_mesh(dx, size):
        """
        Build mesh for computation
        :param dx: float
        Spatial step
        :param size: int
        length of square
        :return: list(np.array, np.array)
        2D Arrays of coordinates
        """
        x = np.arange(-size / 2, size / 2, dx)
        y = np.arange(-size / 2, size / 2, dx)
        x, y = np.meshgrid(x, y)
        return [x, y]

    @staticmethod
    def eval_func(func, mesh):
        """
        Evaluate function on mesh
        :param func: function
        :param mesh: list(np.array, np.array)
        Mesh grid
        :return: 2D np.array
        Values of function on mesh
        """
        x = mesh[0]
        y = mesh[1]
        grid_val = func(x, y)
        return grid_val

    def show_values(self):
        plt3d = plt.figure().gca(projection='3d')
        plt3d.plot_surface(self.mesh[0], self.mesh[1], self.values)
        plt.show()
        plt.close()

    def __update_values__(self):
        """
        Implementation of relaxation method
        :return: 2D array
        Updated values on mesh
        """
        nx = self.mesh[0].shape[0]
        ny = self.mesh[0].shape[1]
        new_grid_val = np.zeros([nx, ny])
        for i in xrange(nx):
            for j in xrange(ny):
                if bool(self.boundary_values[i][j]):
                    new_grid_val[i][j] = self.boundary_values[i][j]
                else:
                    new_grid_val[i][j] = 0.25 * (
                        self.values[i - 1][j] +
                        self.values[i + 1][j] +
                        self.values[i][j - 1] +
                        self.values[i][j + 1]
                    )
        return new_grid_val

    def solve(self, error, show=False):
        """
        Evaluate new values on mesh grid
        :param i: float
        Required error
        :param show: bool
        Plot or not
        :return: 2D np.array
        Finite values on mesh
        """
        i = 0
        pbar = tqdm(total = error)
        while np.linalg.norm(self.values - self.__update_values__())>error:
            pbar.update(1)
            self.values = self.__update_values__()
            i+=1
        pbar.close()
        if show:
            print 'Number of iterations: {}'.format(i)
            plt.figure()
            im = plt.imshow(self.values)
            plt.colorbar(im)
            plt.show()
            plt.close()

        return self.values
