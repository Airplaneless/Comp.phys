import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt

J = 1.0
H = np.array([0.0, 0.0, 1.0])


def spin_interaction(sconfig, i, j):
    idx = i + 1 if i + 1 < sconfig.shape[0] else 0
    jdx = j + 1 if j + 1 < sconfig.shape[1] else 0
    E = -J * np.dot(sconfig[i][j], (sconfig[idx][j] + sconfig[i][jdx] + sconfig[i - 1][j] + sconfig[i][j - 1]))
    E -= np.dot(H, (sconfig[idx][j] + sconfig[i][jdx] + sconfig[i - 1][j] + sconfig[i][j - 1]))
    return E


def energy(sconfig):
    nrj = 0
    S = np.zeros(3)
    for j in range(sconfig.shape[0]):
        for i in range(sconfig.shape[1]):
            idx = i + 1 if i + 1 < sconfig.shape[0] else 0
            jdx = j + 1 if j + 1 < sconfig.shape[1] else 0
            S += sconfig[i][j]
            nrj -= J * np.dot(sconfig[i][j], (sconfig[idx][j] + sconfig[i][jdx]))
    nrj -= np.dot(H, S)
    return nrj


def magnetization(sconfig):
    return np.sum(sconfig) / float(sconfig.shape[0] * sconfig.shape[1])


def metropolis(sconfig, temp):
    mat = sconfig.copy()
    for i in range(sconfig.shape[0]):
        for j in range(sconfig.shape[1]):
            randvec = np.random.rand(1, 3)[0] * ((-1) ** np.random.randint(0, 2)) / 10.0
            e0 = spin_interaction(mat, i, j)
            mat[i][j] += randvec
            mat[j][i] /= np.linalg.norm(mat[j][i])
            e1 = spin_interaction(mat, i, j)
            if (e1 - e0) < 0:
                pass
            else:
                if np.random.rand() <= np.exp(-(e1 - e0) / temp):
                    mat[j][i] -= randvec
    return mat


def spin_structure_factor(lattices, size, axis):
    """
    Evaluate spin structure
    :param lattices: SpinLattice
    :param size: tuple(2)
    size of back? space
    :param axis: str
    'xy' or 'z'
    :return: numpy.ndarray
    Spin structure
    """
    structures = list()
    for k in tqdm(range(len(lattices))):
        structure = np.zeros((lattices[0].shape[0], lattices[0].shape[1]))
        xsum = ysum = zsum = 0.0
        # for each point in lattice
        for n in xrange(size[0]):
            for m in xrange(size[1]):
                vec_q = 4 * np.pi * np.array([n+1, m+1]) / np.array([lattices[0].shape[0], lattices[0].shape[1]])
                vec_q = -2 * np.pi + vec_q
                # summurize by all lattice
                for i in xrange(lattices[0].shape[0]):
                    for j in xrange(lattices[0].shape[1]):
                        xsum += lattices[k][i][j][0] * np.exp(-1j * np.dot(vec_q, np.array([i, j])))
                        ysum += lattices[k][i][j][1] * np.exp(-1j * np.dot(vec_q, np.array([i, j])))
                        zsum += lattices[k][i][j][2] * np.exp(-1j * np.dot(vec_q, np.array([i, j])))
                if axis == 'xy':
                    structure[n][m] = xsum ** 2 + ysum ** 2
                elif axis == 'z':
                    structure[n][m] = zsum ** 2
        structures.append(structure)
    spin_structure = np.mean(structures, axis=0)
    return spin_structure


def run(sconfig, temp, max_epoch, verbose=False):
    mags = list()
    energies = list()
    mean_energy = 0
    mean_mag = 0
    lattices = list()
    new = metropolis(sconfig, temp)
    for t in tqdm(xrange(max_epoch), disable=not verbose):
        a = metropolis(new, temp)
        mags.append(magnetization(a))
        energies.append(energy(a))
        new = a
        lattices.append(new)
    return lattices, energies, mags

if __name__ == '__main__':
    init_spins = np.random.rand(10, 10, 3)
    lattices, E, M = run(init_spins, temp=2.0, max_epoch=1000, verbose=True)
    xy_factor = spin_structure_factor(lattices, axis='xy')
    plt.imshow(xy_factor)
    plt.show()
