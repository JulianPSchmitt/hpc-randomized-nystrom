import numpy as np
from scipy import linalg
import scipy
from mpi4py import MPI
import sketching
import tsqr

np.random.seed(2002)


def rand_nystrom_cholesky(A, Omega, rank):
    """Compute the randomized Nyström rank k approximation given the sketching
    matrix Omega (uses Cholesky decomposition). The method relies on the Nyström
    approximation which incorporates the 'Q' factor of QR decomposition."""
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


def rand_nystrom_cholesky_no_Q(A, Omega, rank):
    """Compute the randomized Nyström rank k approximation given the sketching
    matrix Omega (uses Cholesky decomposition). The method does not use the 'Q'
    factor of the QR decomposition to increase efficency (potentially at the
    cost of less stability)."""
    C = A @ Omega
    B = Omega.T @ C
    L = np.linalg.cholesky(B)
    Z = linalg.solve_triangular(L, C.T, lower=True).T
    R = np.linalg.qr(Z, mode='r')
    if min(R.shape) == rank:
        U_t, Sigma, V_t = linalg.svd(R)
    else:
        U_t, Sigma, V_t = scipy.sparse.linalg.svds(R, k=rank)
    U = Z @ V_t.T @ np.diag(Sigma**(-1))
    return U, np.diag(Sigma**2)


def rand_nystrom_cholesky_parallel(
    A, n: int, l: int, truncate_rank: int, comm: MPI.Comm,
    seed=2002, sketching_mode='BSRHT'
):
    """
    Compute the randomized Nyström rank k approximation of A using Cholesky
    decomposition.

    A is sketched using a 2D grid of p×p processors:
    1) Compute C_ij = A_ij Ω_j and sum reduce row-wise
    2) Compute B_i = Ω_i C_i using the block-row distribution of C and sum
       reduce

    C is kept row-wise distributed after skteching, to reuse the partitioning
    when applying TSQR. For this reason, the Z = C*L^(-1) is computed in
    parallel.

    The method relies on the Nyström approximation which incorporates the 'Q'
    factor of QR decomposition. Therefore, the TSQR explicitely computes Q. This
    increases stability but reduces efficency.

    Finally, the matrices U and S are returned exclusively on the root
    processor.
    """
    rank = comm.Get_rank()
    # Sketch A in parallel using a 2D processor grid
    B, C = sketching.sketch_2D(A, n, l, comm, seed, sketching_mode)

    # Solve a triangular system using row-block distribution of C
    L = np.zeros((l, l))
    if rank == 0:
        L = np.linalg.cholesky(B)
    comm.Bcast(L, root=0)
    Z = linalg.solve_triangular(L, C.T, lower=True).T

    # Execute TSQR
    Q, R = tsqr.tsqr(Z, n, comm, True)

    # Compute SVD sequential
    S = None
    U_t = np.zeros((truncate_rank, l), dtype=float)
    if rank == 0:
        if min(R.shape) == truncate_rank:
            U_t, Sigma, V_t = linalg.svd(R)
        else:
            U_t, Sigma, V_t = scipy.sparse.linalg.svds(R, k=truncate_rank)
        U_t = np.array(U_t)
        S = np.diag(Sigma**2)

    # Exploit the row-block distribution of Q to compute matrix-matrix product
    # in parallel
    comm.Bcast(U_t, root=0)
    if rank != 0:
        U_t = U_t.T
    U_local = Q @ U_t
    U = np.empty((n, truncate_rank))
    comm.Gatherv(U_local, U)
    return U, S


def rand_nystrom_cholesky_no_Q_parallel(
    A, n: int, l: int, truncate_rank: int, comm: MPI.Comm,
    seed=2002, sketching_mode='BSRHT'
):
    """
    Compute the randomized Nyström rank k approximation of A using Cholesky
    decomposition.

    A is sketched using a 2D grid of p×p processors:
    1) Compute C_ij = A_ij Ω_j and sum reduce row-wise
    2) Compute B_i = Ω_i C_i using the block-row distribution of C

    C is kept row-wise distributed after skteching, to reuse the partitioning
    when applying TSQR. For this reason, the Z = C*L^(-1) is computed in
    parallel.

    The method relies on the Nyström approximation which does not need the 'Q'
    factor of QR decomposition. In particular, TSQR does not compute 'Q' to
    increase efficency.

    Finally, the matrices U and S are returned exclusively on the root
    processor.
    """
    rank = comm.Get_rank()
    B, C = sketching.sketch_2D(A, n, l, comm, seed, sketching_mode)
    L = np.zeros((l, l))
    if rank == 0:
        L = np.linalg.cholesky(B)
    comm.Bcast(L, root=0)
    Z = linalg.solve_triangular(L, C.T, lower=True).T
    R = tsqr.tsqr_no_Q(Z, n, comm)

    S = None
    V_t = np.zeros((l, truncate_rank))
    if rank == 0:
        if min(R.shape) == truncate_rank:
            U_t, Sigma, V_t = linalg.svd(R)
        else:
            U_t, Sigma, V_t = scipy.sparse.linalg.svds(R, k=truncate_rank)
        S = np.diag(Sigma**2)
        V_t = V_t.T @ np.diag(Sigma**(-1))
    comm.Bcast(V_t, root=0)
    U_local = Z @ V_t
    U = np.empty((n, truncate_rank))
    comm.Gatherv(U_local, U)
    return U, S
