from mpi4py import MPI
import numpy as np


def tsqr(A, n: int, comm: MPI.Comm, compute_Q=False):
    """Compute the QR decomposition of the matrix A using the TSQR algorithm.
    The method assumes the input matrix A to be row-block distributed on the
    processors of the given communicator. The Q factor will be returned
    distributed row-block wise and the R factor is exclusively computed on the
    root processor."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Compute R
    tree_size = int(np.log2(size))
    Qs = np.empty(tree_size+1, dtype=np.matrix)
    P_new = rank

    Qs[0], R = np.linalg.qr(A)
    for i in range(tree_size):
        if P_new % 2 == 0:
            m, n = R.shape
            R_temp = np.zeros((m, n), dtype='float')
            comm.Recv(R_temp, source=rank + i + 1, tag=44)
            Qs[i+1], R = np.linalg.qr(np.vstack((R, R_temp)))
        else:
            comm.Send(R, dest=rank - i - 1, tag=44)
            break
        P_new = (P_new + 1) // 2

    # Stop computation if Q is not needed
    if compute_Q == False:
        return R

    # Setup for backwards traversal
    active = False
    send = False
    if rank == 0:
        send = True

    # Compute Q
    for i in range(tree_size, 0, -1):
        if not active and rank % 2**(i-1) == 0:
            active = True
        if active and send:
            n = Qs[i-1].shape[1]
            comm.Send(Qs[i][n:2*n, :], dest=rank + i, tag=44)
            Qs[i-1] @= Qs[i][0:n, :]
        elif active:
            n = Qs[i-1].shape[1]
            Qs[i] = np.zeros((n, n), dtype='float')
            comm.Recv(Qs[i], source=rank - i, tag=44)
            Qs[i-1] @= Qs[i]
            send = True

    return Qs[0], R


def tsqr_no_Q(A, n: int, comm: MPI.Comm):
    """Compute the QR decomposition of the matrix A using the TSQR algorithm.
    To exploit additional optimizations, the Q factor is omitted. Note: the R
    factor is only returned on the root processor."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Compute R
    tree_size = int(np.log2(size))
    P_new = rank

    R = np.linalg.qr(A, mode='r')
    for i in range(tree_size):
        if P_new % 2 == 0:
            m, n = R.shape
            R_temp = np.zeros((m, n), dtype='float')
            comm.Recv(R_temp, source=rank + i + 1, tag=44)
            R = np.linalg.qr(np.vstack((R, R_temp)), mode='r')
        else:
            comm.Send(R, dest=rank - i - 1, tag=44)
            break
        P_new = (P_new + 1) // 2

    return R
