""" 
1D Poisson equation with Neumann and Dirichlet boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import glob, re
import scienceplots

import scipy.sparse as sp
import scipy.sparse.linalg as spla



def grid(Lx, Nx):
    """ Create a grid of points. """
    x = np.linspace(0, Lx, Nx+1)
    return x

def Mat(X, neumann_bc, dirichlet_bc):
    """ Assemble the matrix for the Poisson equation. Neumann BC at x = 0 with value neumann_bc. Dirichlet BC at x = 1 with value dirichlet_bc. """
    N = len(X)
    h = X[1] - X[0]
    A = sp.diags([1, -2, 1], [-1, 0, 1], shape=(N, N), format='csc') / h**2
    A[0, 0] = -1.0 / h
    A[0, 1] =  1.0 / h
    A[-1, -1] = 1.0
    A[-1, -2] = 0.0
    # A[0, 1] = 1/h
    A[-1, -1] = 1
    print(A.todense())
    return A

def RHS(X, f, neumann_bc, dirichlet_bc):
    """ Assemble the right-hand side for the Poisson equation. Neumann BC at x = 0 with value neumann_bc. Dirichlet BC at x = 1 with value dirichlet_bc. """
    N = len(X)
    h = X[1] - X[0]
    b = np.zeros(N)
    b[1:-1] = f(X[1:-1])
    b[0] = neumann_bc
    b[-1] = dirichlet_bc
    return b

def SolvePoisson(X, f, neumann_bc, dirichlet_bc):
    """ Solve the Poisson equation. Neumann BC at x = 0 with value neumann_bc. Dirichlet BC at x = 1 with value dirichlet_bc. """
    A = Mat(X, neumann_bc, dirichlet_bc)
    b = RHS(X, f, neumann_bc, dirichlet_bc)
    u = spla.spsolve(A, b)
    return u

def PlotPoisson(X, u, f, neumann_bc, dirichlet_bc):
    """ Plot the solution to the Poisson equation. Neumann BC at x = 0 with value neumann_bc. Dirichlet BC at x = 1 with value dirichlet_bc. """
    plt.plot(X, u, 'r-')
    # plt.plot(X, f(X), 'b-')
    # plt.plot([0, 1], [neumann_bc, dirichlet_bc], 'kx')
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    plt.legend(['$u$', '$f$', 'BC'])

def main():
    """ Main function. """
    Lx = 1
    Nx = 100
    X = grid(Lx, Nx)
    f = lambda x:  10.0
    neumann_bc = -0.0
    dirichlet_bc = 10.0
    u = SolvePoisson(X, f, neumann_bc, dirichlet_bc)
    PlotPoisson(X, u, f, neumann_bc, dirichlet_bc)
    plt.show()

if __name__ == '__main__':
    main()