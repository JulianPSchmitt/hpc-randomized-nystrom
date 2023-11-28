from mpi4py import MPI
import numpy as np


def tsqr(A, n: int, l: int, comm: MPI.Comm, compute_Q=False):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute the row-blocks across the processors
    block_rows = n // size
    submatrix = np.zeros((block_rows, l), dtype='float')
    comm.Scatterv(A, submatrix, root=0)

    # Compute R
    tree_size = int(np.log2(size))
    Qs = np.empty(tree_size+1, dtype=np.matrix)
    P_new = rank

    Qs[0], R = np.linalg.qr(submatrix)
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

    comm.Gatherv(Qs[0], A)
    return A, R
