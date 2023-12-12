from os import environ

# turn off threads in numpy (with openblas)
environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from randomized_nystrom import rand_nystrom_cholesky_parallel
from mpi4py import MPI
from os.path import join
from mnist import build_A
from tqdm import tqdm
from time import time
from __init__ import _FOLDER


def find_runtimes_parallel(
    A,
    comm,
    sketching_mode,
    ls: list[int] = [600, 1000, 2000],
    ks: list[int] = [200, 300, 400, 500, 600],
    seed=2002,
    nRuns: int = 10,
    average_over: str = None,
):
    # This runs through over the values for l and k
    # and averages over "l", "k" or "all"

    m, n = A.shape

    alltimes = np.zeros((len(ls), len(ks), nRuns))
    allmeans = np.zeros((len(ls), len(ks)))
    allstd = np.zeros((len(ls), len(ks)))
    if rank == 0:
        pbar = tqdm(total=len(ls))

    for i, l in enumerate(ls):
        ks_this = [k for k in ks if k <= l]
        for o, k in enumerate(ks_this):
            for j in range(nRuns):
                t1 = time()
                U, S = rand_nystrom_cholesky_parallel(
                    A=A,
                    n=n,
                    l=l,
                    truncate_rank=k,
                    comm=comm,
                    seed=seed,
                    sketching_mode=sketching_mode,
                )

                t2 = time()
                if rank == 0:
                    alltimes[i, o, j] = t2 - t1
            if rank == 0:
                allmeans[i, o] = np.mean(alltimes[i, o, :])
                allstd[i, o] = np.std(alltimes[i, o, :])
        if rank == 0:
            pbar.update(1)

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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    def rbf_kernel(X, n, c):
        X = X[:n, :]
        if rank == 0:
            print(f"Shape: {X.shape}")
        A = build_A(X, c=c, save=False)
        return A

    n = 2**13
    nRuns = 3
    all_As = []
    dataset_names = []
    methods = ["BSRHT", "SASO"]
    method_names = ["SRHT-block", "SASO"]
    ls = [400, 600, 1000, 2000]  # [64, 128, 256]
    ks = [100, 200, 350, 500, 700, 900]  # [15, 30, 60, 120, 240]

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
    if rank == 0:
        print(f"Going to pol with shape: {A_pol.shape}")
    all_As.append(A_pol)
    dataset_names.append("Pol-R10-p1")
    # A_exp = test_matricies[1]
    # all_As.append(A_exp)
    # dataset_names.append("Exp-R10-q0.25")

    # From here we generate plots for all A and all methods
    for i, A in enumerate(all_As):
        for o, method in enumerate(methods):
            all_times, all_std = find_runtimes_parallel(
                A=A,
                comm=comm,
                sketching_mode=method,
                ls=ls,
                ks=ks,
                nRuns=nRuns,
                average_over=None,
            )

            if rank == 0:
                fig, ax = plt.subplots(1, 1)
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
                    f"Parallel runtime for {dataset_names[i]}, Omega from"
                    f" {method_names[o]}"
                )
                plt.xlabel("Approximation rank")
                plt.ylabel("Runtime [s]")
                plt.legend()
                savepath = join(
                    _FOLDER,
                    "plots",
                    f"runtime_par_{dataset_names[i]}_{method_names[o]}.png",
                )
                plt.savefig(savepath)
