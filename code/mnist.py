import numpy as np
import randomized_nystrom as rns
import pandas as pd
rng = np.random.default_rng(seed=2002)


def build_A_sequential(data, c=10**4, save=False):
    '''
    Function to build A out of a data base using the RBF exp( -|| x i - x j || /
    c) Notice that we only need to fill in the upper triangle part of A since
    it's symmetric and its diagonal elements are all 1.
    '''
    n = data.shape[0]
    A = np.zeros((n, n))
    for j in range(n):
        for i in range(j):
            A[i, j] = np.exp(-(np.linalg.norm(data[i, :] - data[j, :])**2)/c)
    A = A + np.transpose(A)
    np.fill_diagonal(A, 1.0)
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


n = 2**11
rank = 200
l = 600
sigma = 10**4

data, labels = read_data("./data/mnist/mnist_780", save=False)
A = build_A_sequential(data, sigma, save=True)
Omega = rns.generate_SRHT(l, n)
U, Sigma = rns.rand_nystrom_cholesky(A, Omega, rank)
A_Nystrom = U @ Sigma @ U.T

print("Nyström via Cholesky:")
print(np.linalg.norm(A-A_Nystrom, ord='nuc')/np.linalg.norm(A, ord='nuc'))
