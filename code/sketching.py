from mpi4py import MPI
import numpy as np
import srht
import saso


def deep_transpose(X, size):
    arrs = np.split(X, size, axis=1)
    # Flatten the sub-arrays
    raveled = [np.ravel(arr) for arr in arrs]
    # Join them back up into a 1D array
    return np.concatenate(raveled)


def sketch_2D(A, n: int, l: int, comm: MPI.Comm, seed=700,  mode='BSRHT'):
    """
    Sketch a n×n matrix A from right and left using BSRHT or SASO.

    The matrix A is distributed on a p×p processor grid and the sketching matrix
    Ω is distributed block row-wise.

    Each local block is sketched, then all results are sum-reduced:
    1) Perform C_ij = A_ij Ω_j and sum reduce all C_ij row-wise
    2) Compute B_i = Ω_i C_i using the block-row distribution of C and sum
       reduce

    Return B and C (B is available on the root processors and C is returned
    distributed row-block wise.)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_blocks = (2 * n) // size

    # Split into a column and a row communicator
    comm_rows = comm.Split(color=rank / 2, key=rank % 2)
    comm_cols = comm.Split(color=rank % 2, key=rank / 2)
    rows_rank = comm_rows.Get_rank()

    # GENERATE OMEGA (sketching matrix)
    if mode == 'BSRHT':
        Omega = srht.block_SRHT(l=l, m=n, comm=comm_cols, seed=seed)
    elif mode == 'SASO':
        Omega = saso.SASO(l=l, m=n, comm=comm_cols, seed=seed)
    else:
        raise NameError("Invalid Mode: " + mode)

    # DISTRIBUTE A
    submatrix = np.zeros((n_blocks, n), dtype='float')
    comm_rows.Scatterv(A, submatrix, root=0)
    block_matrix = np.zeros((n_blocks, n_blocks), dtype='float')
    comm_cols.Scatterv(deep_transpose(
        submatrix, comm_cols.Get_size()), block_matrix, root=0)

    # COMPUTE C
    c_local = block_matrix@Omega.T
    c_sum = np.zeros((n_blocks, l), dtype=float)
    comm_cols.Reduce(c_local, c_sum, op=MPI.SUM, root=rank % 2)

    # DISTRIBUTE C (row-block wise)
    C = np.zeros((n_blocks // 2, l), dtype=float)
    comm_rows.Scatterv(c_sum, C, root=rank/2)

    # COMPUTE B
    c_blocks = n_blocks // 2
    b_local = Omega[:, rows_rank*c_blocks:(rows_rank+1)*c_blocks]@C
    B = np.zeros((l, l))
    comm.Reduce(b_local, B, op=MPI.SUM)

    return B, C
