import numpy as np
import matplotlib.pyplot as plt
from randomized_nystrom import (
    rand_nystrom_cholesky,
)
from os.path import join
from mnist import build_A
from tqdm import tqdm
from srht import fast_SRHT
from time import time
from __init__ import _FOLDER


def find_runtimes(
    A,
    method=np.random.random,
    ls: list[int] = [600, 1000, 2000],
    ks: list[int] = [200, 300, 400, 500, 600],
    nRuns: int = 10,
    average_over: str = None,
):
    # This runs through over the values for l and k
    # and averages over "l", "k" or "all"

    m, n = A.shape

    alltimes = np.zeros((len(ls), len(ks)))

    for i, l in tqdm(enumerate(ls), total=len(ls)):
        ks_this = [k for k in ks if k <= l]
        for o, k in enumerate(ks_this):
            t1 = time()

            for _ in range(nRuns):
                Omega = method((n, l))
                U, S = rand_nystrom_cholesky(A, Omega, rank=k)

            t2 = time()
            alltimes[i, o] = (t2 - t1) / nRuns

    if average_over == "l":
        return np.mean(alltimes, axis=0).flatten()
    elif average_over == "k":
        return np.mean(alltimes, axis=1).flatten()
    elif average_over == "all":
        return np.mean(alltimes, axis=None)
    elif average_over == "none" or average_over is None:
        return alltimes
    else:
        raise Exception("Unknown argument for 'average_over'!")


if __name__ == "__main__":

    def rbf_kernel(X, n, c):
        X = X[:n, :]
        print(f"Shape: {X.shape}")
        A = build_A(X, c=c, save=False)
        return A

    n = 2**10
    all_As = []
    dataset_names = []
    methods = [
        lambda x: fast_SRHT(x[1], x[0]).T
    ]  # , np.random.random # gaussian not always lead to pos. definite matrices...
    method_names = ["SRHT-seq"]  # , "Gaussian"
    ls = [64, 128, 256]
    ks = [16, 32, 64, 128, 256]

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
        "test_matricies_1024.npy",
    )
    test_matricies = np.load(test_matrix_path)
    A_pol = test_matricies[1, 1, 1]
    print(f"Going to pol with shape: {A_pol.shape}")
    all_As.append(A_pol)
    dataset_names.append("Pol-R10-p1")
    # A_exp = test_matricies[2, 1, 1] # Currently exponential does not work with chol.
    # all_As.append(A_exp)
    # dataset_names.append("Exp-R10-q0.25")

    # From here we generate plots for all A and all methods
    for i, A in enumerate(all_As):
        for o, method in enumerate(methods):
            fig, ax = plt.subplots(1, 1)
            all_times = find_runtimes(
                A=A,
                method=method,
                ls=ls,
                ks=ks,
                nRuns=5,
                average_over=None,
            )

            for j, times in enumerate(all_times):
                ks_this = [k for k in ks if k <= ls[j]]
                ax.plot(ks_this, times[: len(ks_this)], label=f"l={ls[j]}")
            plt.title(
                f"Nystrom error {dataset_names[i]}, Omega from"
                f" {method_names[o]}"
            )
            plt.xlabel("Approximation rank")
            plt.ylabel("Runtime [s]")
            plt.legend()
            savepath = join(
                _FOLDER,
                "plots",
                f"runtime_{dataset_names[i]}_{method_names[o]}.png",
            )
            plt.savefig(savepath)
