import numpy as np
import matplotlib.pylab as plt

def make_mesh(dx, size):
    x = np.arange(-size/2, size/2, dx)
    y = np.arange(-size/2, size/2, dx)
    x, y = np.meshgrid(x, y)
    return [x, y]


def eval_func(func, mesh):
    x = mesh[0]
    y = mesh[1]
    grid_val = func(x, y)
    return grid_val


def set_boundary(bound_val, mesh):
    nx = mesh[0].shape[0]
    ny = mesh[0].shape[1]
    z1 = np.zeros([nx, ny])
    z2 = np.ones([nx, ny])
    for i in range(nx):
        for j in range(ny):
            if i==0 or j==0 or i==nx-1 or j==ny-1:
                z1[i][j] = bound_val
                z2[i][j] = 0
    return z1, z2


def iter_calc(mesh, grid_val):
    nx = mesh[0].shape[0]
    ny = mesh[0].shape[1]
    new_grid_val = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            if i==0 or j==0 or i==nx-1 or j==ny-1:
                new_grid_val[i][j] = grid_val[i][j]
            else:
                new_grid_val[i][j] = 0.25*(
                    grid_val[i-1][j]+
                    grid_val[i+1][j]+
                    grid_val[i][j-1]+
                    grid_val[i][j+1]
                )
    return new_grid_val


def solve(i, mesh, grid_val):
    for j in range(i):
        grid_val = iter_calc(mesh, grid_val)
    return grid_val

if __name__ == '__main__':
    f = lambda x,y: -(x**2+y**2)

    mesh = make_mesh(0.1, 10)
    bound, z = set_boundary(10, mesh)
    init_val = eval_func(f, mesh)
    init_val = init_val*z + bound
    res = solve(10, mesh, init_val)

    im = plt.imshow(res)
    plt.colorbar(im)
    plt.show()