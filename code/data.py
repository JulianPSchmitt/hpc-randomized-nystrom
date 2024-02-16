import numpy as np
import randomized_nystrom as rns
import srht
import saso
import pandas as pd
from mpi4py import MPI
from enum import Enum
from dataclasses import dataclass
from omegaconf import OmegaConf
from __init__ import _FOLDER
from os.path import join
from typing import Optional
rng = np.random.default_rng(seed=2002)


def rbf(data: np.ndarray, c: int, savepath: str = None):
    '''
    Function to build A out of a data base using the RBF exp( -|| x i - x j ||^2
    / c^2). The function is solely based on NumPy broadcasting and is therefore
    superior to methods involving Python loops.
    '''
    data_norm = np.sum(data ** 2, axis=-1)
    A = np.exp(-(1/c**2) * (data_norm[:, None] +
               data_norm[None, :] - 2 * np.dot(data, data.T)))

    if savepath is not None:
        A.tofile(savepath, sep=',', format='%10.f')

    return A


def read_mnist(filename: str, size: int = 784, savepath: str = None):
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
        # The first digit is the label
        labels[i] = int(l[0])
        l = l[2:]
        indices_values = [tuple(map(float, pair.split(':')))
                          for pair in l.split()]
        # Separate indices and values
        indices, values = zip(*indices_values)
        indices = [int(i) for i in indices]
        # Fill in values at the specified indices
        data[i, indices] = values

    if savepath is not None:
        data.tofile('./denseData.csv', sep=',', format='%10.f')
        labels.tofile('./labels.csv', sep=',', format='%10.f')

    return data, labels


def read_yearPredictionMSD(filename: str, size: int = 784, savepath: str = None):
    '''
    Read YearPredictionMSD sparse data from filename and transforms this into a
    dense matrix, each line representing an entry of the database (i.e. a
    "flattened" image)
    '''
    dataR = pd.read_csv(filename, sep=',', header=None)
    n = len(dataR)
    data = np.zeros((n, size))
    labels = np.zeros((n, 1))

    # Format
    for i in range(n):
        l = dataR.iloc[i, 0]
        # The first digit is the label
        labels[i] = int(l[0])
        l = l[6:]
        indices_values = [tuple(map(float, pair.split(':')))
                          for pair in l.split()]
        # Separate indices and values
        indices, values = zip(*indices_values)
        indices = [int(i) for i in indices]
        # Fill in values at the specified indices
        data[i, indices] = values

    if savepath is not None:
        data.tofile('./denseData.csv', sep=',', format='%10.f')
        labels.tofile('./labels.csv', sep=',', format='%10.f')

    return data, labels


def pol_decay(n: int, r: int, p: int = 1):
    A = [1.0 for _ in range(r)] + [(2.0 + o) ** (-p) for o in range(n - r)]
    return np.diag(A)


def exp_decay(n: int, r: int, q: float = 0.25):
    A = [1.0 for _ in range(r)] + [(10.0) ** (-(o + 1)*q)
                                   for o in range(n - r)]
    return np.diag(A)


class Dataset(Enum):
    MNIST = 0
    YEAR_PREDICTION_MSD = 1
    EXP_DECAY = 2
    POLY_DECAY = 3


class Variant(Enum):
    CHOLESKY = 0
    SVD = 1


class Sketch(Enum):
    SRHT = 0
    SASO = 1


@dataclass
class Config():
    """
    Dataclass to store cofiguration from CLI.
    """
    dataset: Dataset = Dataset.MNIST
    input_dim: int = 2048
    sketch_dim: int = 400
    truncate_rank: int = 200
    rbf_smooth: Optional[int] = 100
    effective_rank: Optional[int] = 10
    data_decay: Optional[float] = 1
    variant: Variant = Variant.CHOLESKY
    sketch: Sketch = Sketch.SRHT


if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Read CLI Args
    args = Config()
    config = OmegaConf.merge(OmegaConf.structured(args), OmegaConf.from_cli())
    print("Running with following configuration:\n", config)

    # Build Matrix A
    A = None
    if rank % 2 == 0:
        if config.dataset == Dataset.POLY_DECAY:
            A = pol_decay(config.input_dim,
                          config.effective_rank, config.data_decay)
        elif config.dataset == Dataset.EXP_DECAY:
            A = exp_decay(config.input_dim,
                          config.effective_rank, config.data_decay)
        elif config.dataset == Dataset.MNIST:
            data, labels = read_mnist(
                join(_FOLDER, "data", "mnist", "mnist_780"))
            A = rbf(data, config.rbf_smooth)
        else:
            data, labels = read_yearPredictionMSD(
                join(_FOLDER, "data", "yearPredictionMSD", "year"))
            A = rbf(data, config.rbf_smooth)

    # Compute Nyström Approximation sequentially
    if size == 1:
        # Create Sketching-Matrix
        if config.sketch == Sketch.SRHT:
            Omega = srht.fast_SRHT(config.sketch_dim, config.input_dim).T
        else:
            Omega = saso.SASO(config.sketch_dim, config.input_dim, comm).T

        # Approximate A
        if config.variant == Variant.CHOLESKY:
            U, Sigma = rns.rand_nystrom_cholesky(
                A, Omega, config.truncate_rank)
        else:
            U, Sigma = rns.rand_nystrom_svd(A, Omega, config.truncate_rank)

    # Compute Nyström Approximation in parallel
    else:
        # Determine Sketching Mode
        if config.sketch == Sketch.SRHT:
            sketching_mode = "BRSHT"
        else:
            sketching_mode = "SASO"

        # Approximate A
        if config.variant == Variant.CHOLESKY:
            U, Sigma = rns.rand_nystrom_cholesky_parallel(
                A, config.input_dim, config.sketch_dim, config.truncate_rank,
                comm, sketching_mode=sketching_mode)
        else:
            U, Sigma = rns.rand_nystrom_svd_parallel(
                A, config.input_dim, config.sketch_dim, config.truncate_rank,
                comm, sketching_mode=sketching_mode)

    # Print Results
    if rank == 0:
        A_Nystrom = U @ Sigma @ U.T
        approx_error = np.linalg.norm(
            A-A_Nystrom, ord='nuc') / np.linalg.norm(A, ord='nuc')
        print("Approximation Error: ", approx_error, "\n")
