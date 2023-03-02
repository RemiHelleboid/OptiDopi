import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import glob, re
import scienceplots

import scipy.sparse as sp
import scipy.sparse.linalg as spla




# plt.style.use(["science", "high-vis", "grid"])



""" POISSON 2D SOLVER WITH NEUMANN AND DIRICHLET BOUNDARY CONDITIONS """


def create_grid(xmin, xmax, ymin, ymax, nx, ny):
    Nx = nx + 2
    Ny = ny + 2
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    X = np.linspace(xmin - dx, xmax + dx, Nx)
    Y = np.linspace(ymin - dy, ymax + dy, Ny)
    X, Y = np.meshgrid(X, Y)

    # Types points (0: border, 1: corner, 2: ghost, 3: inner)
    TypesPoints = np.zeros((Nx, Ny))
    TypesPoints[0, :] = 2 
    TypesPoints[-1, :] = 2
    TypesPoints[:, 0] = 2
    TypesPoints[:, -1] = 2
    TypesPoints[1, 1] = 1
    TypesPoints[1, -2] = 1
    TypesPoints[-2, 1] = 1
    TypesPoints[-2, -2] = 1
    TypesPoints[2:-2, 2:-2] = 3

    # im = plt.imshow(TypesPoints)
    # plt.colorbar(im, ticks=[0, 1, 2, 3])
    # plt.show()


    FlatTypesPoints = TypesPoints.flatten()
    indices_borders = np.where(FlatTypesPoints == 0)[0]
    indices_corners = np.where(FlatTypesPoints == 1)[0]
    indices_ghost = np.where(FlatTypesPoints == 2)[0]
    indices_inner = np.where(FlatTypesPoints == 3)[0]

    return X, Y, indices_borders, indices_corners, indices_ghost, indices_inner


def plot_grid(xmin, xmax, ymin, ymax, nx, ny):
    """ Plot the grid """
    X, Y, indices_borders, indices_corners, indices_ghost, indices_inner = create_grid(xmin, xmax, ymin, ymax, nx, ny)
    fig, ax = plt.subplots()
    print(indices_borders)
    xplot = X.flatten()
    yplot = Y.flatten()
    ax.plot(xplot[indices_borders], yplot[indices_borders], 'o', color='black')
    ax.plot(xplot[indices_corners], yplot[indices_corners], '*', color='red')
    ax.plot(xplot[indices_ghost], yplot[indices_ghost], '+', color='blue')
    ax.plot(xplot[indices_inner], yplot[indices_inner], 'v', color='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.legend(['borders', 'corners', 'ghost', 'inner'])
    fig.tight_layout()
    plt.show()



def poisson_matrix_2D(X, Y, indices_borders):
    """ Create the matrix of the Poisson equation (Neumann BC all over) """
    nx, ny = X.shape
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    A = sp.diags([-1/dx**2, -1/dy**2, 2/dx**2 + 2/dy**2, -1/dx**2, -1/dy**2], [-nx, -1, 0, 1, nx], shape=(nx*ny, nx*ny))
    A = A.tocsr()
    A.eliminate_zeros()
    A[indices_borders, :] = 0
    A[indices_borders, indices_borders] = sp.eye(len(indices_borders))
    return A


def poisson_rhs_2D(X, Y, indices_borders, VborderNeuman, ScalarForcingTerm):
    """ Create the right hand side of the Poisson equation (Neuman BC all over) """
    nx, ny = X.shape
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    b = np.zeros(nx*ny)
    b[indices_borders] = VborderNeuman
    b[1:-1, 1:-1] = ScalarForcingTerm(X[1:-1, 1:-1], Y[1:-1, 1:-1])
    return b


