# turn off threads in numpy (with openblas)
from os import environ

environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from randomized_nystrom import (
    rand_nystrom_cholesky_parallel,
    rand_nystrom_svd_parallel,
)
from mpi4py import MPI
from os.path import join
from mnist import build_A
from tqdm import tqdm
from __init__ import _FOLDER


def relative_error_rank_parallel(
    A: np.ndarray,
    n: int,
    l: int,
    ks: list[int],
    comm: MPI.Comm,
    sketching_mode: str,
    cholesky: bool = False,
    seed: int = 2002,
    ax=None,
    normAnuc=None,
    **plt_kwargs,
):
    rank = comm.Get_rank()
    if rank == 0 and normAnuc is None:
        normAnuc = norm(A, "nuc")
    errs = [None for _ in range(len(ks))]
    if rank == 0:
        pbar = tqdm(total=len(ks))
    for i, k in enumerate(ks):
        if cholesky:
            U, S = rand_nystrom_cholesky_parallel(
                A,
                n=n,
                l=l,
                truncate_rank=k,
                comm=comm,
                seed=seed,
                sketching_mode=sketching_mode,
            )
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
            errs[i] = norm(A - A_nyst, "nuc") / normAnuc
            pbar.update(1)

    if rank == 0:
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(ks, errs, marker="o", **plt_kwargs)
        ax.set_yscale("log")
        ax.set_xlabel("Approximation rank")
        ax.set_ylabel("Relative error, nuclear norm")

        return ax


def compare_relative_error_sketching_parallel(
    A,
    n: int,
    comm: MPI.Comm,
    sketching_mode: str,
    ls: list[int] = [600, 1000, 2000],
    ks: list[int] = [200, 300, 400, 500, 600],
    cholesky: bool = True,
    seed: int = 2002,
    ax=None,
    title="Nystrom RBF approximation",
    savepath=None,
    **plotting_kwargs,
):
    rank = comm.Get_rank()
    if ax is None and rank == 0:
        fig, ax = plt.subplots(1, 1)

    m, n = A.shape
    if rank == 0:
        # Only calculate this once, as it takes a long time...
        normAnuc = norm(A, "nuc")
    else:
        normAnuc = None

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

        ks_this = [k for k in ks if k <= l]
        ax = relative_error_rank_parallel(
            A=A,
            n=n,
            l=l,
            ks=ks_this,
            comm=comm,
            sketching_mode=sketching_mode,
            cholesky=cholesky,
            seed=seed,
            ax=ax,
            normAnuc=normAnuc,
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

    def rbf_kernel(X, n, c):
        X = X[:n, :]
        if rank == 0:
            print(f"Shape: {X.shape}")
        A = build_A(X, c=c, save=False)
        return A

    n = 2**13  # Should be 10**12 or higher when running final time!
    ls = [400, 600, 1000, 2000]  # [64, 128, 256]
    ks = [100, 200, 350, 500, 700, 900]  # [15, 30, 60, 120, 240]
    all_As = []
    dataset_names = []
    methods = ["BSRHT", "SASO"]
    method_names = ["SRHT-block", "SASO"]

    # Mnist dataset
    MNIST_X_path = join(_FOLDER, "data", "MNISTtrain.npy")
    X = np.load(MNIST_X_path)
    A = rbf_kernel(X=X, n=n, c=1e4)
    all_As.append(A)
    dataset_names.append("RBF-MNIST")

    # YearMSD dataset
    YEAR_X_path = join(_FOLDER, "data", "YearMSD.npy")
    X = np.load(YEAR_X_path)
    A = rbf_kernel(X=X, n=n, c=(1e4) ** 2)
    all_As.append(A)
    dataset_names.append("RBF-YearMSD-1e4")

    A = rbf_kernel(X=X, n=n, c=(1e5) ** 2)
    all_As.append(A)
    dataset_names.append("RBF-YearMSD-1e5")

    # Synthetic datasets
    test_matrix_path = join(
        _FOLDER,
        "data",
        "test_matricies_8192.npy",
    )
    test_matricies = np.load(test_matrix_path)
    A_pol = test_matricies[0]
    all_As.append(A_pol)
    dataset_names.append("Pol-R10-p1")
    A_exp = test_matricies[1]
    all_As.append(A_exp)
    dataset_names.append("Exp-R10-q0.25")

    # From here we generate plots for all A and all methods
    for i, A in enumerate(all_As):
        for o, method in enumerate(methods):
            if dataset_names[i] == "Exp-R10-q0.25":
                cholesky = False
            else:
                cholesky = True

            ax = compare_relative_error_sketching_parallel(
                A,
                n=n,
                comm=comm,
                sketching_mode=method,
                ls=ls,  # [600, 1000]
                ks=ks,  # [16, 32, 64, 128, 256],  # [200, 400, 600]
                cholesky=cholesky,
                title=(
                    f"Nystrom error {dataset_names[i]}, Omega from"
                    f" {method_names[o]}"
                ),
                savepath=join(
                    _FOLDER,
                    "plots",
                    f"relerror_{dataset_names[i]}_{method_names[o]}.png",
                ),
            )
            plt.show(block=False)
