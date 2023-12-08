import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from os.path import join
from scipy.linalg import hadamard


# From Rokhlin, Szlam, Tygert paper:
# "A Randomized Algorithm For Principal Component Analysis"
# Currently only used for this plot, but should be moved if used elsewhere
def generate_matrix(m, sig_kp1=0.1, k=10):
    U = hadamard(m, dtype="d") / np.sqrt(m)
    V = hadamard(2 * m, dtype="d") / np.sqrt(2 * m)
    firstSig = [sig_kp1 ** (np.floor(j / 2) / 5) for j in range(1, k + 1)]
    sigmas = firstSig + [
        sig_kp1 * (m - j) / (m - 11) for j in range(k + 1, m + 1)
    ]
    S = np.concatenate([np.diag(sigmas), np.zeros((m, m), dtype="d")], axis=1)

    return U @ S @ V.T, sigmas


# Needed for plot here, but if used for anything else
# than plot, should be moved to proper file.
def randomized_svd(
    A: np.ndarray,
    k: int = 10,  # Desired rank
    p: int = 6,  # p=6 gives prob 0.99
    returnQ1: bool = False,
    seed: int = 42,
):
    np.random.seed(seed)

    m, n = A.shape
    l = p + k

    Omega1 = np.random.random((n, l))

    Y = (A @ A.T) @ A @ Omega1
    Q1 = np.linalg.qr(Y)[0]

    B = Q1.T @ A

    if min(B.shape) == k:
        U_t, Sigma, V_t = scipy.linalg.svd(B)
    else:
        U_t, Sigma, V_t = scipy.sparse.linalg.svds(B, k=k)

    # Truncate
    U_t = U_t[:, :k]
    Sigma = Sigma[:k]
    V_t = V_t[:, :k]
    # Also apply Q1 to U_t
    U = Q1 @ U_t

    if returnQ1:
        return U, Sigma, V_t, Q1
    else:
        return U, Sigma, V_t


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
