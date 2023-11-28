import numpy as np
from mpi4py import MPI


def SASO(l: int, m: int, comm: MPI.Comm, seed=2002):
    """
    Create a sketching matrix Omega with dimensions l x m in parallel where l is
    the sketching dimension. Omega is a Short-axis-sparse sketching operator.
    Following current research results, we set the sparsity parameter k to 8.
    The method can handle multiple processors, connected using the given "comm".
    Note that after returning, each processor will have a different submatrix of
    Omega of size l x (m/p).
    """
    p = comm.Get_size()
    rank = comm.Get_rank()

    r = m / p
    k = np.min([8, l])
    if np.ceil(r) != r:
        print(
            f"Warning! r is not a power of 2! Consider padding the data with"
            f" 0-values"
        )
    else:
        r = int(r)

    seed = comm.bcast(seed, root=0)
    np.random.seed(seed + rank + 100)

    # Nonzero values in short-axis vectors are iid ~ Rademacher
    # i.e. samples uniformly from the disjoint intervals (-2, 1], [1, 2)
    # (this protects against the possibility of the vector being orthogonal to
    # a column of the matrix to be sketched)
    Omega = np.zeros((l, r), dtype=float)
    for i in range(r):
        choice = np.random.choice(l, k, replace=False)
        signs = np.random.choice([-1, 1], k, replace=True)
        values = signs * np.random.uniform(1, 2, k)
        Omega[choice, i] = values

    return Omega


# run using (here 4 processors):
# mpiexec -n 4 python srht.py
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    n = 2**12
    l = 600

    r1 = SASO(l=l, m=n, comm=comm)

    if rank == 0:
        print(r1[:5, :5])
