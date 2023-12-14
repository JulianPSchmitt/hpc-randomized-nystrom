import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from randomized_nystrom import rand_nystrom_cholesky, rand_nystrom_svd
from os.path import join
from mnist import build_A
from tqdm import tqdm
from srht import fast_SRHT
from __init__ import _FOLDER


def relative_error_rank(
    A, Omega, ks, cholesky=False, ax=None, normAnuc=None, **plt_kwargs
):
    if normAnuc is None:
        normAnuc = norm(A, "nuc")

    errs = [None for _ in range(len(ks))]
    for i, k in tqdm(enumerate(ks), total=len(ks)):
        if cholesky:
            U, S = rand_nystrom_cholesky(A, Omega, rank=k)
        else:
            U, S = rand_nystrom_svd(A, Omega, rank=k)
        A_nyst = U @ S @ U.T

        errs[i] = norm(A - A_nyst, "nuc") / normAnuc

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.plot(ks, errs, marker="o", **plt_kwargs)
    ax.set_yscale("log")
    ax.set_xlabel("Approximation rank")
    ax.set_ylabel("Relative error, nuclear norm")

    return ax


def compare_relative_error_sketching(
    A,
    method=np.random.random,
    ls=[600, 1000, 2000],
    ks=[200, 300, 400, 500, 600],
    cholesky: bool = True,
    ax=None,
    title="Nystrom RBF approximation",
    savepath=None,
    **plotting_kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    m, n = A.shape

    normAnuc = norm(A, "nuc")

    colors = plotting_kwargs.get("color", None)
    if colors is None and len(ls) <= 5:
        colors = "krbgy"
    labels = plotting_kwargs.get("label", None)
    if not labels is None:
        assert len(labels) == len(ls)

    for i, l in tqdm(enumerate(ls), total=len(ls)):
        Omega = method((n, l))

        if not labels is None:
            plotting_kwargs.update({"label": f"l={l}"})
        else:
            plotting_kwargs.update({"label": f"l={l}"})

        if not colors is None:
            plotting_kwargs.update({"color": colors[i]})

        ks_this = [k for k in ks if k <= l]
        ax = relative_error_rank(
            A=A,
            Omega=Omega,
            ks=ks_this,
            cholesky=cholesky,
            ax=ax,
            normAnuc=normAnuc,
            **plotting_kwargs,
        )

    ax.set_title(title)
    if "label" in plotting_kwargs:
        plt.legend()
    if not savepath is None:
        plt.savefig(savepath)
    return ax


if __name__ == "__main__":

    def rbf_kernel(X, n, c):
        X = X[:n, :]
        print(f"Shape: {X.shape}")
        A = build_A(X, c=c, save=False)
        return A

    n = 2**13
    ls = [400, 600, 1000, 2000]  # [64, 128, 256]
    ks = [100, 200, 350, 500, 700, 900]  # [15, 30, 60, 120, 240]
    all_As = []
    dataset_names = []
    methods = [lambda x: fast_SRHT(x[1], x[0]).T]
    method_names = ["SRHT-seq"]

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
    print(f"Going to pol with shape: {A_pol.shape}")
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

            ax = compare_relative_error_sketching(
                A,
                method=method,
                ls=ls,
                ks=ks,
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
