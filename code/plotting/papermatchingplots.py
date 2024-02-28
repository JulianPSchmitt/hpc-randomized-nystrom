import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from randomized_nystrom import rand_nystrom_cholesky
from data import rbf
from os.path import join


def errorplot(X: np.ndarray, n: int, c: float, ls: list[int], ks: list[int]):
    X = X[:n, :]
    print(f"Shape: {X.shape}")
    A = rbf(X, c=c, savepath=None)

    colors = "krbgy"

    errs = [[None for _ in range(len(ks))] for _ in range(len(ls))]

    for o, l in tqdm(enumerate(ls), total=len(ls)):
        for i, k in enumerate(ks):
            Omega = np.random.random((n, l))

            U, S = rand_nystrom_cholesky(A=A, Omega=Omega, rank=l)
            A_nyst = U @ S @ U.T

            errs[o][i] = np.linalg.norm(A - A_nyst, "nuc") / np.linalg.norm(
                A, "nuc"
            )

    # Error plots!
    plt.figure(figsize=(8, 6), dpi=80)
    for i in range(len(ls)):
        plt.plot(
            ks,
            errs[o],
            c=colors[i],  # "#003aff",
            marker="o",
            label=f"l={ls[i]}",
        )
    plt.yscale("log")
    plt.legend()
    plt.title(
        f"Nystrom RBF approximation ($\sigma=10^{int(np.log10(c))}$), "
        + r"$\phi\left( \| x i - x j \| \right) = e^{- \| x i - x j \|^2 /"
        r" \sigma^2}$"
    )
    plt.xlabel("Approximation rank")
    plt.ylabel("Relative error, nuclear norm")


def paperMatchingPlots(
    savepath: str,
    MNIST_X_path: str,
    YearMSD_X_path: str,
    n: int = 2**10,
    plotTitleSpecifier: str = "",
):
    try:
        X = np.load(MNIST_X_path)
        ls = [600, 1000, 2000]
        ks = [200, 300, 400, 500, 600]

        # MNIST plot
        errorplot(X=X, n=n, c=100**2, ls=ls, ks=ks)
        plt.savefig(
            join(
                savepath,
                "MNIST_trace_relative_error" + plotTitleSpecifier + ".png",
            )
        )
        plt.show()
    except Exception as err:
        print(
            f"Was unable to create the MNIST error plot does to error: {err}"
        )

    try:
        X2 = np.load(YearMSD_X_path)
        ls = [600, 1000, 2000, 2500, 3000]
        ks = [100, 200, 500, 1000]
        errorplot(X=X2, n=n, c=(1e4) ** 2, ls=ls, ks=ks)
        plt.savefig(
            join(
                savepath,
                "YearMSD_10_4_trace_relative_error"
                + plotTitleSpecifier
                + ".png",
            )
        )
        plt.show()
    except Exception as err:
        print(
            "Was unable to create the YearMSD 1e-4 error plot does to error:"
            f" {err}"
        )

    try:
        X2 = np.load(YearMSD_X_path)
        ls = [600, 1000, 2000, 2500, 3000]
        ks = [100, 200, 500, 1000]
        errorplot(X=X2, n=n, c=(1e5) ** 2, ls=ls, ks=ks)
        plt.savefig(
            join(
                savepath,
                "YearMSD_10_5_trace_relative_error"
                + plotTitleSpecifier
                + ".png",
            )
        )
        plt.show()
    except Exception as err:
        print(
            "Was unable to create the YearMSD 1e-5 error plot does to error:"
            f" {err}"
        )
