import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from tqdm import tqdm
from copy import deepcopy

INTERACTION_ENERGY = 1.0


class SpinLattice:
    """
    Implementation of spin lattice
    """
    def __init__(self, matrix):
        """
        SpinLattice constructor
        :param matrix: np.ndarray
        Matrix of spins
        """
        self.matrix = matrix

    def __copy__(self):
        return SpinLattice(self.matrix)

    @staticmethod
    def build1d_random(x_size, y_size):
        """
        Build random lattice of of 1d spins for Ising model
        :param x_size: float
        Size of lattice
        :param y_size: float
        Size of lattice
        :return: SpinLattice
        """
        matrix = np.random.randint(low=0, high=2, size=(x_size, y_size))
        return SpinLattice(1 - 2*matrix)

    @staticmethod
    def build3d_random(x_size, y_size):
        """
        Build random lattice of of 3d spins for Heisenberg model
        :param x_size: float
        Size of lattice
        :param y_size: float
        Size of lattice
        :return: SpinLattice
        """
        matrix = np.random.rand(x_size, y_size, 3) - np.random.rand(x_size, y_size, 3)
        for i in xrange(len(matrix)):
            for j in xrange(len(matrix[i])):
                matrix[i][j] = matrix[i][j] / np.sqrt(np.dot(matrix[i][j], matrix[i][j]))
        return SpinLattice(np.random.rand(x_size, y_size, 3))

    def energy(self, H):
        """
        Evaluate energy of spin lattice
        :param H: 3-element np.array
        Vector of magnetic field
        :return: float
        Energy of spin configuration
        """
        res = 0
        if H:
            spins_sum = np.zeros(3)
        for j in range(self.matrix.shape[0]):
            for i in range(self.matrix.shape[1]):
                idx = i + 1 if i + 1 < self.matrix.shape[0] else 0
                jdx = j + 1 if j + 1 < self.matrix.shape[1] else 0
                if H:
                    spins_sum += self.matrix[i][j]
                res -= INTERACTION_ENERGY * np.dot(self.matrix[i][j], (self.matrix[idx][j] + self.matrix[i][jdx]))
        if H:
            res -= np.dot(H, spins_sum)
        return res

    def magnetization(self):
        """
        Evaluate magnetization
        :return: float
        Magnetization
        """
        return np.sum(self.matrix) / float(self.matrix.shape[0] * self.matrix.shape[1])

    def show(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if len(self.matrix[0][0]) == 1:
            ax.imshow(self.matrix)
            plt.show()
        else:
            raise ValueError("Can't visualize")


class SpinLatticeList:
    """
    Implementation of list DS for SpinLatticeS
    """
    def __init__(self, SpinLattices):
        """
        SpinLatticeList constructor
        :param SpinLattices: list of SpinLattice
        """
        self.list = SpinLattices

    def get(self):
        return self.list

    def magnetization(self, start, stop):
        """
        Magnetisations on interval
        :param start: int
        Begin of interval
        :param stop: int
        End of interval
        :return: float[]
        """
        return [lattice.magnetization() for lattice in self.list[start:stop]]

    def energy(self, H, start, stop):
        """
        Energys on interval
        :param H: 3-element np.array
        Vector of magnetic field
        :param start: int
        Begin of interval
        :param stop: int
        End of interval
        :return: float[]
        """
        return [lattice.energy(H) for lattice in self.list[start:stop]]

    def visualize(self, start, stop):
        fig = plt.figure(figsize=(6, 6))
        plt.title('Iteration = 0')
        im = plt.imshow(self.list[0].matrix, interpolation='bilinear')

        def update(j, *args, **kwargs):
            im.set_array(self.list[j].matrix)
            plt.title('Iteration = {}'.format(j))
            return im,

        return animation.FuncAnimation(fig, update, frames=range(start, stop), blit=True)


class SimulationRunner:

    def __init__(self, model, temperature, magnetic_field, epoch):
        """
        SimulationRunner constructor
        :param model: str
        'Ising' or 'Hberg'
        :param interaction_energy: float
        Energy of spin-spin interation
        :param temperature: float
        Temperature in energy unit
        :param magnetic_field: 3-element np.ndarray
        Vector of magnetic field
        :param epoch: int
        Number of Monte-Carlo steps
        """
        self.model = model
        self.temperature = temperature
        self.magnetic_field = magnetic_field
        self.epoch = epoch

    @staticmethod
    def spin_structure(lattices, size):
        """
        Evaluate spin structure for XY or Z
        :param lattices: list of SpinLattice
        :param size: tuple(2)
        size of back space
        :return: list of numpy.ndarray
        List of spin structures for xy and z axes
        """
        structures_xy = np.zeros((size[0], size[1]))
        structures_z = np.zeros((size[0], size[1]))
        x_size, y_size, vec_dim = np.shape(lattices.list[0].matrix)
        for k in tqdm(xrange(len(lattices.list)), desc='Evaluate spin structure'):
            structure_xy = np.zeros((size[0], size[1]))
            structure_z = np.zeros((size[0], size[1]))
            xsum = ysum = zsum = 0.0
            # for each point in lattice
            for I in xrange(size[0]):
                for J in xrange(size[1]):
                    vec_q = 10 * np.pi * np.array([I+1, J+1]).astype('float')
                    vec_q /= np.array([size[0], size[1]]).astype('float')
                    vec_q += - 10 * np.pi / 2.0
                    # summurize by all lattice
                    for i in xrange(x_size):
                        for j in xrange(y_size):
                            vec_r = np.array([i+1, j+1]).astype('float')
                            xsum += lattices.list[k].matrix[i][j][0] * np.exp(-1j * np.dot(vec_q, vec_r))
                            ysum += lattices.list[k].matrix[i][j][1] * np.exp(-1j * np.dot(vec_q, vec_r))
                            zsum += lattices.list[k].matrix[i][j][2] * np.exp(-1j * np.dot(vec_q, vec_r))
                    structure_xy[I][J] = xsum ** 2 + ysum ** 2
                    structure_z[I][J] = zsum ** 2
                structures_xy += structure_xy
                structures_z += structure_z
        return structures_xy, structures_z

    def _spin_interaction(self, lattice, i, j):
        """
        Evaluate spin interaction for (i,j)
        :param lattice: SpinLattice
        :param i: int
        :param j: int
        :return: float
        """
        idx = i + 1 if i + 1 < lattice.matrix.shape[0] else 0
        jdx = j + 1 if j + 1 < lattice.matrix.shape[1] else 0
        E = -INTERACTION_ENERGY * np.dot(
            lattice.matrix[i][j],
            lattice.matrix[idx][j] + lattice.matrix[i][jdx] +
            lattice.matrix[i - 1][j] + lattice.matrix[i][j - 1]
        )
        if self.magnetic_field:
            E -= np.dot(
                self.magnetic_field,
                lattice.matrix[idx][j] + lattice.matrix[i][jdx] +
                lattice.matrix[i - 1][j] + lattice.matrix[i][j - 1]
            )
        return E

    def _make_step(self, old_lattice):
        """
        Perform one step of metropolis algorithm
        :param old_lattice: SpinLattice
        Old spin lattice
        :return: SpinLattice
        Spin lattice with updated spins
        """
        lattice = deepcopy(old_lattice)
        if self.model == 'Hberg':
            for i in range(lattice.matrix.shape[0]):
                for j in range(lattice.matrix.shape[1]):
                    vec_rand = np.random.rand(1, 3)[0] * ((-1) ** np.random.randint(0, 2)) / 10.0
                    e0 = self._spin_interaction(lattice, i, j)
                    lattice.matrix[i][j] += vec_rand
                    lattice.matrix[j][i] /= np.linalg.norm(lattice.matrix[j][i])
                    e1 = self._spin_interaction(lattice, i, j)
                    if e1 > e0 and np.random.rand() > np.exp(-(e1 - e0) / self.temperature):
                        lattice.matrix[j][i] -= vec_rand
            return lattice
        elif self.model == 'Ising':
            for i in range(lattice.matrix.shape[0]):
                for j in range(lattice.matrix.shape[1]):
                    e0 = self._spin_interaction(lattice, i, j)
                    lattice.matrix[i][j] *= -1.0
                    e1 = self._spin_interaction(lattice, i, j)
                    rand = np.random.random()
                    prob = np.exp(-(e1 - e0) / self.temperature)
                    if e1 > e0 and rand > prob:
                        lattice.matrix[i][j] *= -1.0
            return lattice
        else:
            raise ValueError("{} model not implemented".format(self.model))

    def run(self, lattice, verbose=True):
        """
        Run modeling
        :param lattice: SpinLattice
        Initial spin configuration
        :param verbose: bool
        Enable tqdm load bar
        :return: SpinLatticeList
        List of SpinLattices
        """
        res = [lattice]
        for t in tqdm(xrange(self.epoch), desc='Running', disable=not verbose):
            new = self._make_step(res[-1])
            res.append(new)
        return SpinLatticeList(res)
