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


def AssembleMatrixPoisson_2D(L, Nxy, Dirichlet_bc):
    """ Assemble the matrix for the 2D Poisson equation. Dirichlet BC at x = 0, x = Lx, y = 0, y = Ly with value Dirichlet_bc. """
    Nx, Ny = Nxy
    hx, hy = L[0]/Nx, L[1]/Ny
    # Two-dimensional Laplacian matrix with 5-point stencil
    A = sp.diags([1, 1, -4, 1, 1], [-Nx, -1, 0, 1, Nx], shape=(Nx*Ny, Nx*Ny), format='csc') / hx**2
    A = A + sp.diags([1, 1, -4, 1, 1], [-Nx, -1, 0, 1, Nx], shape=(Nx*Ny, Nx*Ny), format='csc') / hy**2
    # Dirichlet BC at x = 0, x = Lx, y = 0, y = Ly
    A[0, 0] = 1.0
    A[0, 1] = 0.0
    A[0, Nx] = 0.0
    A[Nx-1, Nx-1] = 1.0
    A[Nx-1, Nx-2] = 0.0
    A[Nx-1, 2*Nx-1] = 0.0
    A[Nx*(Ny-1), Nx*(Ny-1)] = 1.0
    A[Nx*(Ny-1), Nx*(Ny-1)-1] = 0.0
    A[Nx*(Ny-1), Nx*(Ny-2)-1] = 0.0
    A[Nx*Ny-1, Nx*Ny-1] = 1.0
    A[Nx*Ny-1, Nx*Ny-2] = 0.0
    A[Nx*Ny-1, Nx*Ny-Nx-1] = 0.0
    for i in range(1, Nx-1):
        A[i, i] = 1.0
        A[i, i-1] = 0.0
        A[i, i+1] = 0.0
        A[i, i+Nx] = 0.0
        A[Nx*(Ny-1)+i, Nx*(Ny-1)+i] = 1.0
        A[Nx*(Ny-1)+i, Nx*(Ny-1)+i-1] = 0.0
        A[Nx*(Ny-1)+i, Nx*(Ny-1)+i+1] = 0.0
        A[Nx*(Ny-1)+i, Nx*(Ny-2)+i] = 0.0
    for j in range(1, Ny-1):
        A[Nx*j, Nx*j] = 1.0
        A[Nx*j, Nx*j-1] = 0.0
        A[Nx*j, Nx*j+1] = 0.0
        A[Nx*j, Nx*(j+1)] = 0.0
        A[Nx*(j+1)-1, Nx*(j+1)-1] = 1.0
        A[Nx*(j+1)-1, Nx*(j+1)-2] = 0.0
        A[Nx*(j+1)-1, Nx*(j+1)-1-Nx] = 0.0
        A[Nx*(j+1)-1, Nx*j-1] = 0.0
    return A

def fxy(x, y):
    """ Right-hand side function. """
    return -1.0

def AssembleRHS_2D(L, Nxy, fxy, dirichlet_bc):
    """ Assemble the right-hand side vector for the 2D Poisson equation. """
    Nx, Ny = Nxy
    hx, hy = L[0]/Nx, L[1]/Ny
    b = np.zeros(Nx*Ny)
    for j in range(Ny):
        for i in range(Nx):
            b[j*Nx+i] = fxy(i*hx, j*hy)
    # Dirichlet BC at x = 0, x = Lx, y = 0, y = Ly
    for j in range(Ny):
        b[j*Nx] = dirichlet_bc
        b[j*Nx+Nx-1] = dirichlet_bc
    for i in range(Nx):
        b[i] = dirichlet_bc
        b[Nx*(Ny-1)+i] = dirichlet_bc
    return b



if __name__ == "__main__":
    Lx, Ly = 1.0, 1.0
    Nx, Ny = 10, 10
    X, Y = grid(Lx, Ly, Nx-1, Ny-1)
    Dirichlet_bc = 1.0
    A = AssembleMatrixPoisson_2D([Lx, Ly], [Nx, Ny], Dirichlet_bc)
    B = AssembleRHS_2D([Lx, Ly], [Nx, Ny], fxy, Dirichlet_bc)
    U = spla.spsolve(A, B)
    U = np.reshape(U, (Ny, Nx))
    plt.figure()
    plt.contourf(X, Y, U, 100)
    plt.colorbar()
    plt.show()
