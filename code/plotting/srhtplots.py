import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
from srht import fast_SRHT as srht
from os.path import join
from typing import Optional


# Utility function to get the Least squares estimate
def realLS(W, b, returnResidual: bool = False):
    res = np.linalg.lstsq(W, b, rcond=None)
    if returnResidual:
        return res[0], np.sqrt(np.sum(res[1]))
    else:
        return res[0]


# Utility to get the value of l
def bestL(eps: float, n: int, m: int):
    # finding l
    l = (
        np.ceil(eps ** (-2) * np.log(n))
        * (np.sqrt(n) + np.sqrt(np.log(m))) ** 2
    )
    # print(f"l would have been {l}")
    if m < np.ceil(l):
        print(f"Warning! l > m, so all columns are used (for eps={eps})")
    l = int(min(np.ceil(l), m))
    return l


def srht_plot(
    W: np.ndarray,
    b: np.ndarray,
    savepath: str,
    nRuns: int = 10,
    l: Optional[int] = None,
    epsilons: list[float] = [100.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1],
    plotTitleSpecifier: str = "",
):
    epsilons = np.array(epsilons)
    timeFull = np.array([0.0 for _ in range(len(epsilons))])
    timeComp = np.array([0.0 for _ in range(len(epsilons))])
    resFull = np.array([0.0 for _ in range(len(epsilons))])
    resComp = np.array([0.0 for _ in range(len(epsilons))])
    relErrW = np.array([0.0 for _ in range(len(epsilons))])
    relErrRes = np.array([0.0 for _ in range(len(epsilons))])

    for i, eps in tqdm(enumerate(epsilons), total=len(epsilons)):
        for _ in range(nRuns):
            t1 = time()
            realx, errFull = realLS(W, b, returnResidual=True)
            t2 = time()
            m, n = W.shape

            # Find Omega
            if l == None:
                goodL = bestL(eps=eps, n=n, m=m)
            else:
                goodL = l
            omega = srht(goodL, m)
            # Find error in sketched space
            omegaW = omega @ W
            omegab = omega @ b
            estx, errComp = realLS(omegaW, omegab, returnResidual=True)

            t3 = time()
            timeFull[i] += t2 - t1
            timeComp[i] += t3 - t2

            resFull[i] += errFull  # Best possible error
            resComp[i] += errComp  # Error in Omega-space
            relErrW[i] += abs(
                np.linalg.norm(omegaW) - np.linalg.norm(W)
            ) / np.linalg.norm(W)
            relErrRes[i] += (
                np.linalg.norm(W @ estx - b) / errFull
            )  # Relative error in real space

    # Reduce the sums to averages
    timeFull = timeFull / nRuns
    timeComp = timeComp / nRuns
    resFull = resFull / nRuns
    resComp = resComp / nRuns
    relErrW = relErrW / nRuns
    relErrRes = relErrRes / nRuns

    ### Lots of plots!!!
    # Time
    plt.figure(figsize=(8, 6), dpi=80)
    plt.loglog(
        epsilons, timeFull, c="#003aff", marker="o", label="Full problem"
    )
    plt.loglog(
        epsilons, timeComp, c="#00b310", marker="*", label="Compressed problem"
    )
    plt.legend()
    plt.title(r"$\varepsilon$" + ", time taken to build and compute")
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel("Time, s")
    plt.savefig(join(savepath, "TimeBuiltEps" + plotTitleSpecifier + ".png"))
    # Norm of residual
    plt.figure(figsize=(8, 6), dpi=80)
    plt.loglog(
        epsilons, resFull, c="#003aff", marker="o", label="In full problem"
    )
    plt.loglog(
        epsilons,
        resComp,
        c="#00b310",
        marker="*",
        label="In compressed problem",
    )
    plt.legend()
    plt.title(r"$\varepsilon$" + ", norm of residual")
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel("Norm of residual")
    plt.savefig(
        join(savepath, "NormResidualEps" + plotTitleSpecifier + ".png")
    )
    # Relative error in spectral norm
    plt.figure(figsize=(8, 6), dpi=80)
    plt.loglog(
        epsilons, relErrW, c="#5400b3", marker="o", label="Relative error"
    )
    plt.loglog(
        epsilons,
        epsilons,
        c="#676b74",
        linestyle="dashed",
        label=r"$\varepsilon$ as stated by Thm 2, week 6",
    )
    plt.legend()
    plt.title(
        r"$\varepsilon$"
        + ", relative error spectral norm "
        + r"$ | \| \Omega A\| 2 - \|A\| 2 |/\ | A\| 2$"
    )
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$ | \| \Omega A\| 2 - \|A\| 2 |/\ | A\| 2$")
    # Relative error (estimate of epsilon)
    print(f"Testing relErrRes: {relErrRes}")
    plt.savefig(join(savepath, "RelResidualEps" + plotTitleSpecifier + ".png"))
    plt.figure(figsize=(8, 6), dpi=80)
    plt.loglog(
        epsilons,
        relErrRes - 1,
        c="#5400b3",
        marker="o",
        label="Relative error",
    )
    plt.loglog(
        epsilons,
        epsilons,
        c="#676b74",
        linestyle="dashed",
        label=r"$\varepsilon$ as stated by first equation in week 7",
    )
    plt.legend()
    plt.title(
        r"$\varepsilon$"
        + ", relative residual spectral norm "
        + r"$ \| A x_{srht} - b \|_{2} /\ \| A x - b \|_{2} - 1 \approx"
        r" \varepsilon$"
    )
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$ \| A x_{srht} - b \|_{2} /\ \| A x - b \|_{2} - 1$")
    plt.savefig(
        join(savepath, "EstimateEpsFromRes" + plotTitleSpecifier + ".png")
    )
    plt.show()
