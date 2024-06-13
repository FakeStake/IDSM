from numba import njit
import numpy as np

@njit
def lu_decomposition(a):
    n = a.shape[0]
    L = np.zeros_like(a)
    U = np.zeros_like(a)
    
    for i in range(n):
        L[i, i] = 1  # Diagonal elements of L are 1
        for j in range(i, n):
            U[i, j] = a[i, j] - np.dot(L[i, :i], U[:i, j])
        for j in range(i+1, n):
            L[j, i] = (a[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
    
    return L, U

@njit
def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

@njit
def back_substitution(U, y):
    n = U.shape[0]
    x = np.zeros_like(y)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

@njit
def lin_solve(a, b):
    L, U = lu_decomposition(a)
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    return x