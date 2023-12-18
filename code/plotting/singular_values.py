import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
from randomized_nystrom import (
    rand_nystrom_cholesky_parallel,
    rand_nystrom_svd_parallel
)
from mpi4py import MPI
from os.path import join
import mnist
from scipy.sparse.linalg import svds
from tqdm import tqdm
from __init__ import _FOLDER
from os import environ
# turn off threads in numpy (with openblas)
environ["OMP_NUM_THREADS"] = "1"


def poly_decay(R, n, p):
    '''
    Build a nxn matrix A such that
    A = diag(1, .., 1, 2^(-p), ..., (n-R+1)^(-p)
    '''
    diagA = np.ones(n)
    diagA[R:] = [(i)**(-p) for i in range(2, n-R+2)]
    A = np.diag(diagA)
    return A


def exp_decay(R, n, q):
    '''
    Build a nxn matrix A such that
    A = diag(1, ..., 1, 10^(-q), ..., 10^(-(n-R)q)
    '''
    diagA = np.ones(n)
    diagA[R:] = [10**(-i*q) for i in range(1, n-R+1)]
    A = np.diag(diagA)
    return A


def read_data_year(filename, size=784, save=False):
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
        l = l[6:]
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


def relative_error_rank_parallel(
    A: np.ndarray,
    n: int,
    l: int,
    k: int,
    A_singular,
    Cholesky,
    comm: MPI.Comm,
    sketching_mode: str,
    seed: int = 2002,
    ax=None,
    **plt_kwargs,
):
    rank = comm.Get_rank()
    errs = None

    if Cholesky:
        U, S = rand_nystrom_cholesky_parallel(
            A,
            n=n,
            l=l,
            truncate_rank=k,
            comm=comm,
            seed=seed,
            sketching_mode=sketching_mode,
        )
        if rank == 0:
            A_nyst = U @ S @ U.T
            A_nyst_singular = - \
                np.sort(-svds(A_nyst, k, return_singular_vectors=False))
            errs = A_nyst_singular / A_singular
    else:
        U, S = rand_nystrom_svd_parallel(
            A,
            n=n,
            l=l,
            truncate_rank=k,
            comm=comm,
            seed=seed,
            sketching_mode=sketching_mode,
        )
        if rank == 0:
            A_nyst = U @ S @ U.T
            A_nyst_singular = np.linalg.svd(A, compute_uv=False)
            errs = A_nyst_singular[:k] / A_singular

    if rank == 0:
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(range(len(errs)), errs, **plt_kwargs)
        # ax.set_yscale("log")
        ax.set_xlabel("i")
        ax.set_ylabel("$\sigma_i([\![ A_{Nystr} ]\!]_k)/\sigma_i(A)$")

        return ax


def compare_relative_error_sketching_parallel(
    A,
    n: int,
    comm: MPI.Comm,
    sketching_mode: str,
    A_singular,
    Cholesky=True,
    ls: list[int] = [600, 1000, 2000],
    k: int = 400,
    seed: int = 2002,
    ax=None,
    title="Nystrom RBF approximation",
    savepath=None,
    **plotting_kwargs,
):
    rank = comm.Get_rank()
    if ax is None and rank == 0:
        fig, ax = plt.subplots(1, 1)

    colors = plotting_kwargs.get("color", None)
    if colors is None and len(ls) <= 5:
        colors = "krbgy"
    labels = plotting_kwargs.get("label", None)
    if not labels is None:
        assert len(labels) == len(ls)

    if rank == 0:
        pbar = tqdm(total=len(ls))

    for i, l in enumerate(ls):
        if not labels is None:
            plotting_kwargs.update({"label": f"l={l}"})
        else:
            plotting_kwargs.update({"label": f"l={l}"})

        if not colors is None:
            plotting_kwargs.update({"color": colors[i]})

        ax = relative_error_rank_parallel(
            A=A,
            n=n,
            l=l,
            k=k,
            A_singular=A_singular,
            Cholesky=Cholesky,
            comm=comm,
            sketching_mode=sketching_mode,
            seed=seed,
            ax=ax,
            **plotting_kwargs,
        )
        if rank == 0:
            pbar.update(1)

    if rank == 0:
        ax.set_title(title)
        if "label" in plotting_kwargs:
            plt.legend()
        if not savepath is None:
            plt.tight_layout()
            plt.savefig(savepath)
        return ax


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    n = 2**13  # Should be 10**12 or higher when running final time!
    ls = [400, 600, 1000, 2000]
    k = 400
    all_As = []
    dataset_names = []
    methods = ["BSRHT", "SASO"]
    method_names = ["SRHT-block", "SASO"]

    # Mnist dataset
    A = None
    if rank % 2 == 0:
        MNIST_X_path = join(_FOLDER, "data", "mnist", "mnist_780")
        data, labels = mnist.read_data(MNIST_X_path, save=False)
        A = mnist.build_A(data, 10**4, save=False)
    all_As.append(A)
    dataset_names.append("RBF-MNIST")

    # YearMSD dataset
    A = None
    if rank % 2 == 0:
        data, labels = read_data_year(
            "./data/YearPredictionMSD/year_780", save=False)
        A = mnist.build_A(data, (1e4) ** 2, save=False)
    all_As.append(A)
    dataset_names.append("RBF-YearMSD-1e4")

    if rank % 2 == 0:
        A = mnist.build_A(data, (1e5) ** 2, save=False)
    all_As.append(A)
    dataset_names.append("RBF-YearMSD-1e5")

    # Synthetic datasets
    A_pol = None
    if rank % 2 == 0:
        A_pol = poly_decay(10, 8192, 1)
    if rank == 0:
        print(f"Going to pol with shape: {A_pol.shape}")
    all_As.append(A_pol)
    dataset_names.append("Pol-R10-p1")

    # From here we generate plots for all A and all methods
    for i, A in enumerate(all_As):
        if rank == 0:
            # Only calculate this once, as it takes a long time...
            A_singular = - \
                np.sort(-svds(A, k, return_singular_vectors=False))
        else:
            A_singular = None
        for o, method in enumerate(methods):
            ax = compare_relative_error_sketching_parallel(
                A,
                n=n,
                comm=comm,
                sketching_mode=method,
                A_singular=A_singular,
                Cholesky=True,
                ls=ls,
                k=k,
                title=(
                    f"Singular value error {dataset_names[i]}, Omega from"
                    f" {method_names[o]}"
                ),
                savepath=join(
                    _FOLDER,
                    "plots",
                    f"singular_values_{dataset_names[i]}_{method_names[o]}.pdf",
                ),
            )
            plt.show(block=False)

    A_exp = None
    if rank % 2 == 0:
        A_exp = exp_decay(10, n, 0.25)

    A_singular = None
    if rank == 0:
        # Only calculate this once, as it takes a long time...
        A_singular = - np.sort(-svds(A_exp, k, return_singular_vectors=False))

    for o, method in enumerate(methods):
        ax = compare_relative_error_sketching_parallel(
            A_exp,
            n=n,
            comm=comm,
            sketching_mode=method,
            A_singular=A_singular,
            Cholesky=False,
            ls=ls,
            k=k,
            title=(
                f"Singular value error Exp-R10-q0.25, Omega from"
                f" {method_names[o]}"
            ),
            savepath=join(
                _FOLDER,
                "plots",
                f"singular_values_Exp-R10-q0.25_{method_names[o]}.pdf",
            ),
        )
        plt.show(block=False)
