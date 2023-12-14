from os import environ

# turn off threads in numpy (with openblas)
environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from randomized_nystrom import (
    rand_nystrom_cholesky,
    rand_nystrom_svd
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
    cholesky=True,
    ls: list[int] = [600, 1000, 2000],
    ks: list[int] = [200, 300, 400, 500, 600],
    nRuns: int = 10,
    average_over: str = None,
):
    # This runs through over the values for l and k
    # and averages over "l", "k" or "all"

    m, n = A.shape

    alltimes = np.zeros((len(ls), len(ks), nRuns))
    allmeans = np.zeros((len(ls), len(ks)))
    allstd = np.zeros((len(ls), len(ks)))

    for i, l in tqdm(enumerate(ls), total=len(ls)):
        ks_this = [k for k in ks if k <= l]
        for o, k in enumerate(ks_this):
            for j in range(nRuns):
                if cholesky:
                    t1 = time()
                    Omega = method((n, l))
                    U, S = rand_nystrom_cholesky(A, Omega, rank=k)
                    t2 = time()
                else:
                    t1 = time()
                    Omega = method((n, l))
                    U, S = rand_nystrom_svd(A, Omega, rank=k)
                    t2 = time()
                alltimes[i, o, j] = t2 - t1
            allmeans[i, o] = np.mean(alltimes[i, o, :])
            allstd[i, o] = np.std(alltimes[i, o, :])

    if average_over == "l":
        return (
            np.mean(allmeans, axis=0).flatten(),
            np.mean(allstd, axis=0).flatten(),
        )
    elif average_over == "k":
        return (
            np.mean(allmeans, axis=1).flatten(),
            np.mean(allstd, axis=1).flatten(),
        )
    elif average_over == "all":
        return np.mean(allmeans, axis=None), np.mean(allstd, axis=None)
    elif average_over == "none" or average_over is None:
        return allmeans, allstd
    else:
        raise Exception("Unknown argument for 'average_over'!")


if __name__ == "__main__":

    def rbf_kernel(X, n, c):
        X = X[:n, :]
        print(f"Shape: {X.shape}")
        A = build_A(X, c=c, save=False)
        return A

    n = 2**13
    nRuns = 10
    As_cholesky = []
    As_svd = []
    dataset_names_cholesky = []
    dataset_names_svd = []
    methods = [
        lambda x: fast_SRHT(x[1], x[0]).T
    ]  # , np.random.random # gaussian not always lead to pos. definite matrices...
    method_names = ["SRHT-seq"]  # , "Gaussian"
    ls = [400, 600, 1000, 2000]  # [64, 128, 256]
    ks = [100, 200, 350, 500, 700, 900]  # [15, 30, 60, 120, 240]

    # Mnist dataset
    MNIST_X_path = join(_FOLDER, "data", "MNISTtrain.npy")
    X = np.load(MNIST_X_path)
    A = rbf_kernel(X=X, n=n, c=1e4)
    As_cholesky.append(A)
    dataset_names_cholesky.append("RBF-MNIST")

    # YearMSD dataset
    YEAR_X_path = join(_FOLDER, "data", "YearMSD.npy")
    X = np.load(YEAR_X_path)
    A = rbf_kernel(X=X, n=n, c=(1e4) ** 2)
    As_cholesky.append(A)
    dataset_names_cholesky.append("RBF-YearMSD-1e4")

    A = rbf_kernel(X=X, n=n, c=(1e5) ** 2)
    As_cholesky.append(A)
    dataset_names_cholesky.append("RBF-YearMSD-1e5")

    # Synthetic datasets
    test_matrix_path = join(
        _FOLDER,
        "data",
        "test_matricies_8192.npy",
    )
    test_matricies = np.load(test_matrix_path)
    A_pol = test_matricies[0]
    print(f"Going to pol with shape: {A_pol.shape}")
    As_cholesky.append(A_pol)
    dataset_names_cholesky.append("Pol-R10-p1")
    A_exp = test_matricies[1] # Currently exponential does not work with chol.
    As_svd.append(A_exp)
    dataset_names_svd.append("Exp-R10-q0.25")

    # From here we generate plots for all A and all methods
    for i, A in enumerate(As_cholesky):
        for o, method in enumerate(methods):
            fig, ax = plt.subplots(1, 1)
            all_times, all_std = find_runtimes(
                A=A,
                method=method,
                cholesky=True,
                ls=ls,
                ks=ks,
                nRuns=nRuns,
                average_over=None,
            )

            for j, times in enumerate(all_times):
                stds = all_std[j]
                ks_this = [k + j - 1 for k in ks if k <= ls[j]]
                ax.errorbar(
                    ks_this,
                    times[: len(ks_this)],
                    yerr=stds[: len(ks_this)],
                    fmt="-",
                    label=f"l={ls[j]}",
                    c="krbgy"[j],
                )
            plt.title(
                f"Sequential runtime for {dataset_names_cholesky[i]}, Omega from"
                f" {method_names[o]}"
            )
            plt.xlabel("Approximation rank")
            plt.ylabel("Runtime [s]")
            plt.legend()
            savepath = join(
                _FOLDER,
                "plots",
                f"runtime_{dataset_names_cholesky[i]}_{method_names[o]}.png",
            )
            plt.savefig(savepath)

    # From here we generate plots for all A via SVD and all methods
    for i, A in enumerate(As_svd):
        for o, method in enumerate(methods):
            fig, ax = plt.subplots(1, 1)
            all_times, all_std = find_runtimes(
                A=A,
                method=method,
                cholesky=False,
                ls=ls,
                ks=ks,
                nRuns=nRuns,
                average_over=None,
            )

            for j, times in enumerate(all_times):
                stds = all_std[j]
                ks_this = [k + j - 1 for k in ks if k <= ls[j]]
                ax.errorbar(
                    ks_this,
                    times[: len(ks_this)],
                    yerr=stds[: len(ks_this)],
                    fmt="-",
                    label=f"l={ls[j]}",
                    c="krbgy"[j],
                )
            plt.title(
                f"Sequential runtime for {dataset_names_svd[i]}, Omega from"
                f" {method_names[o]}"
            )
            plt.xlabel("Approximation rank")
            plt.ylabel("Runtime [s]")
            plt.legend()
            savepath = join(
                _FOLDER,
                "plots",
                f"runtime_{dataset_names_svd[i]}_{method_names[o]}.png",
            )
            plt.savefig(savepath)
