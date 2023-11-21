import numpy as np
from scipy.linalg import hadamard as sp_hadamard
from mpi4py import MPI


# Also using random choice, but using scipy hadamard
def fast_SRHT(l, m, seed=2002):
    np.random.seed(seed)
    d = np.array([1 if np.random.random() < 0.5 else -1 for i in range(m)])
    choice = np.random.choice(range(m), l, replace=False)
    omega = np.sqrt(m / l) * sp_hadamard(m, dtype="d") / np.sqrt(m)

    return omega[choice, :] * d.reshape([1, -1])


def block_SRHT(l, m, comm: MPI.Comm, seed=2002):
    p = comm.Get_size()
    rank = comm.Get_rank()

    r = m / p
    if np.ceil(r) != r:
        print(
            f"Warning! r is not a power of 2! Consider padding the data with"
            f" 0-values"
        )
    else:
        r = int(r)

    seed = comm.bcast(seed, root=0)
    np.random.seed(seed + rank + 100)
    # Different D-matricies for all processors
    d1 = np.array([1 if np.random.random() < 0.5 else -1 for i in range(l)])
    d2 = np.array([1 if np.random.random() < 0.5 else -1 for i in range(r)])
    np.random.seed(seed)
    # Same choice of permutation for all processors
    choice = np.random.choice(range(r), l, replace=False)
    omega = np.sqrt(r / l) * sp_hadamard(r, dtype="d") / np.sqrt(r)

    return d1.reshape([-1, 1]) * omega[choice, :] * d2.reshape([1, -1])


# run using (here 4 processors):
# mpiexec -n 4 python srht.py
if __name__ == "__main__":
    eps = 1.0

    n = 2**12
    l = 600

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    r1 = block_SRHT(l=l, m=n, comm=comm)

    if rank == 0:
        print(r1[:5, :5])
