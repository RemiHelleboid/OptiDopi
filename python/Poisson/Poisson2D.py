import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import glob, re
import scienceplots

import scipy.sparse as sp
import scipy.sparse.linalg as spla



""" Two-dimensional Poisson equation solver with Neuman and Dirichlet boundary conditions. """


def grid(Lx, Ly, Nx, Ny):
    """ Create a grid of points. """
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    return np.meshgrid(x, y)


def AssembleMatrixPoisson(X, Y, indicesNeumann : list, 