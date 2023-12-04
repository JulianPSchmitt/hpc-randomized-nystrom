import numpy as np
import randomized_nystrom as rns
import srht
import pandas as pd
from mpi4py import MPI
rng = np.random.default_rng(seed=2002)


def build_A(data, c=10**4, save=False):
    '''
    Function to build A out of a data base using the RBF exp( -|| x i - x j || /
    c). The function solely based on NumPy broadcasting and is therefore
    superior to methods involving Python loops.
    '''
    data_norm = np.sum(data ** 2, axis=-1)
    A = np.exp(-(1/c) * (data_norm[:, None] +
               data_norm[None, :] - 2 * np.dot(data, data.T)))
    if save:
        A.tofile('./A.csv', sep=',', format='%10.f')
    return A


def read_data(filename, size=784, save=False):
    '''
    Read MNIST sparse data from filename and transforms this into a dense
    matrix, each line representing an entry of the database (i.e. a "flattened"
    image)
    '''
    dataR = pd.read_csv(filename, sep=',', header=None)
    n = len(dataR)
    data = np.zeros((n, size))
    labels = np.zeros((n, 1))
    # Format accordingly
    for i in range(n):
        l = dataR.iloc[i, 0]
        # We know that the first digit is the label
        labels[i] = int(l[0])
        l = l[2:]
        indices_values = [tuple(map(float, pair.split(':')))
                          for pair in l.split()]
        # Separate indices and values
        indices, values = zip(*indices_values)
        indices = [int(i) for i in indices]
        # Fill in the values at the specified indices
        data[i, indices] = values
    if save:
        data.tofile('./denseData.csv', sep=',', format='%10.f')
        labels.tofile('./labels.csv', sep=',', format='%10.f')
    return data, labels


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = 2**11
    truncate_rank = 200
    l = 400
    sigma = 10**4

    A = None
    if rank % 2 == 0:
        data, labels = read_data("./data/mnist/mnist_780", save=False)
        A = build_A(data, sigma, save=False)

    if size == 1:
        Omega = srht.fast_SRHT(l, n).T
        U, Sigma = rns.rand_nystrom_cholesky(A, Omega, truncate_rank)
        print("Sequential")
    else:
        U, Sigma = rns.rand_nystrom_cholesky_parallel(
            A, n, l, truncate_rank, comm)
        print("Parallel")

    if rank == 0:
        A_Nystrom = U @ Sigma @ U.T
        print("Nyström via Cholesky:")
        print(np.linalg.norm(A-A_Nystrom, ord='nuc') /
              np.linalg.norm(A, ord='nuc'))
