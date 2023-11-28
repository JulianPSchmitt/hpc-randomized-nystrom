import numpy as np
from scipy import linalg
import scipy
from mpi4py import MPI
import sketching
import tsqr

np.random.seed(2002)


def rand_nystrom_cholesky(A, Omega, rank):
    """Compute the randomized Nyström rank k approximation given the sketching
    matrix Omega (uses Cholesky decomposition)"""
    C = A @ Omega
    B = Omega.T @ C
    L = np.linalg.cholesky(B)
    Z = linalg.solve_triangular(L, C.T, lower=True).T
    Q, R = np.linalg.qr(Z)
    if min(R.shape) == rank:
        U_t, Sigma, V_t = linalg.svd(R)
    else:
        U_t, Sigma, V_t = scipy.sparse.linalg.svds(R, k=rank)
    U = Q @ U_t
    return U, np.diag(Sigma**2)


def rand_nystrom_cholesky_parallel(
    A, n: int, l: int, truncate_rank: int, comm: MPI.Comm, seed=2002
):
    """Compute the randomized Nyström rank k approximation given the sketching
    matrix Omega (uses Cholesky decomposition)"""
    rank = comm.Get_rank()

    B, C = sketching.sketch_2D_BSRHT(A, n, l, comm, seed)

    Z = None
    if rank == 0:
        L = np.linalg.cholesky(B)
        Z = linalg.solve_triangular(L, C.T, lower=True).T
    Q, R = tsqr.tsqr(Z, n, l, comm, True)

    U = S = None
    if rank == 0:
        if min(R.shape) == truncate_rank:
            U_t, Sigma, V_t = linalg.svd(R)
        else:
            U_t, Sigma, V_t = scipy.sparse.linalg.svds(R, k=truncate_rank)
        U = Q @ U_t
        S = np.diag(Sigma**2)
    return U, S
