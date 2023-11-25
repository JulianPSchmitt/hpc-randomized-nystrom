import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join

# Following imports should be from different file
# Simply here to get plotting started
from plotting import generate_matrix, randomized_svd


def singvalplots(
    savepath: str,
    m: int = 2**11,
    n: int = 2**12,
    k: int = 10,
    p: int = 6,
    sig_kp1s: list[float] = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
    plotTitleSpecifier: str = "",
):
    sig_kp1s = np.array(sig_kp1s)
    errTheorem = [None for _ in range(len(sig_kp1s))]

    plt.figure(figsize=(8, 6), dpi=80)
    for i, sig_kp1 in tqdm(enumerate(sig_kp1s), total=len(sig_kp1s)):
        A, sigmas = generate_matrix(m=m, sig_kp1=sig_kp1, k=k)
        U, s, V, Q1 = randomized_svd(A, k=k, p=p, returnQ1=True)

        errTheorem[i] = np.linalg.norm(A - Q1 @ Q1.T @ A)

        # Plot the decay of the singular values
        plt.loglog(
            np.arange(m),
            sigmas,
            marker="o",
            label=(r"$\sigma_{k+1} = $" + str(sig_kp1)),
        )  # , c="#0800ff"
    # Aux things for plot
    plt.title("Decay on singular values")
    plt.legend()
    plt.xlabel("k")
    plt.ylabel(r"$\sigma_{k}$")
    plt.savefig(
        join(savepath, "DecaySingularVals" + plotTitleSpecifier + ".png")
    )

    # Plot error from theorem
    plt.figure(figsize=(8, 6), dpi=80)
    plt.loglog(
        sig_kp1s,
        errTheorem,
        marker="o",
        label="$ | \| A - Q_1 Q_1^{T}A\| $",
    )  # c="#ff8f00",
    plt.loglog(
        sig_kp1s,
        (1.0 + 11.0 * np.sqrt(k + p) * np.sqrt(min(m, n))) * sig_kp1s,
        marker="o",
        label="$(1 + 11 * \sqrt{k + p} * \sqrt{min(m, n)})\sigma_{k+1}$",
    )
    plt.loglog(
        sig_kp1s,
        (1.0 + 4 * np.sqrt(k + p) / (p - 1) * np.sqrt(min(m, n))) * sig_kp1s,
        marker="o",
        label="$(1 + 4/(p-1) * \sqrt{k + p} * \sqrt{min(m, n)})\sigma_{k+1}$",
    )
    plt.title(r"$ | \| A - Q_1 Q_1^{T}A\| $" + " and decay on singular values")
    plt.legend()
    plt.xlabel(r"$\sigma_{k+1}$")
    plt.ylabel(r"$ | \| A - Q_1 Q_1^{T}A\| $")
    plt.savefig(
        join(
            savepath, "DecaySingularValsAndError" + plotTitleSpecifier + ".png"
        )
    )
    plt.show()
