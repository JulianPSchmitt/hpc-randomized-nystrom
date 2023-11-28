from mpi4py import MPI
import numpy as np
import srht


def sketch_2D_BSRHT(A, n: int, l: int, comm: MPI.Comm, seed=700):
    """
    Sketch a n×n matrix A from right and left using BSRHT.

    The matrix A is distributed on a Pr×Pc processor grid and the SRHT sketching
    matrix Ω is distributed block row-wise.

    Each local block is sketched, then all results are sum-reduced:
    1) Perform C_ij = A_ij Ω_j and sum all C_ij for a given i
    2) Perform B_i = Ω_i C_i and sum all B_i

    Return B and C
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_blocks = (2 * n) // size

    # Get the matrix transposed (ugly computations is necessary since MPI cannot
    # work with NumPy array shapes)
    A_transpose = None
    if rank % 2 == 0:
        arrs = np.split(A, n, axis=1)
        raveled = [np.ravel(arr) for arr in arrs]
        A_transpose = np.concatenate(raveled)

    # Split into a column and a row communicator
    comm_cols = comm.Split(color=rank / 2, key=rank % 2)
    comm_rows = comm.Split(color=rank % 2, key=rank / 2)
    rank_cols = comm_cols.Get_rank()

    # GENERATE OMEGA (sketching matrices)
    Omega_right = srht.block_SRHT(l=l, m=n, comm=comm_cols, seed=seed).T
    Omega_left = srht.block_SRHT(l=l, m=n, comm=comm_rows, seed=seed)

    # DISTRIBUTE A
    # We start by scattering the columns of A
    submatrix = np.empty((n_blocks, n), dtype='float')
    receive_mat = np.empty((n_blocks*n), dtype='float')
    comm_cols.Scatterv(A_transpose, receive_mat, root=0)
    sub_arrs = np.split(receive_mat, n_blocks)
    raveled = [np.ravel(arr, order='F') for arr in sub_arrs]
    submatrix = np.ravel(raveled, order='F')
    # Then we scatter the rows
    block_matrix = np.empty((n_blocks, n_blocks), dtype='float')
    comm_rows.Scatterv(submatrix, block_matrix, root=0)

    # COMPUTE C
    c_local = block_matrix@Omega_right
    c_row = np.empty((n_blocks, l), dtype=float)
    comm_cols.Allreduce(c_local, c_row, op=MPI.SUM)

    # Gather rows of C together
    C = None
    if rank_cols == 0:
        C = np.empty((n, l), dtype=float)
        comm_rows.Gatherv(c_row, C, root=0)

    # COMPUTE B
    b_local = Omega_left@c_row
    B = np.empty((l, l))
    comm_rows.Reduce(b_local, B, op=MPI.SUM, root=0)

    return B, C
