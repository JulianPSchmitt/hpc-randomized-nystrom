from scipy.linalg import hadamard
import numpy as np
from typing import Optional
from scipy.linalg import ldl


def generate_matrix(m, sig_kp1=0.1, k=10):
    U = hadamard(m, dtype="d") / np.sqrt(m)
    V = hadamard(2 * m, dtype="d") / np.sqrt(2 * m)
    firstSig = [sig_kp1 ** (np.floor(j / 2) / 5) for j in range(1, k + 1)]
    sigmas = firstSig + [
        sig_kp1 * (m - j) / (m - 11) for j in range(k + 1, m + 1)
    ]
    S = np.concatenate([np.diag(sigmas), np.zeros((m, m), dtype="d")], axis=1)

    return U @ S @ V.T, sigmas


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

    # svdalg = TruncatedSVD(k, random_state=seed)
    # res = svdalg.fit(B)
    # V = res.components_
    U, s, V = np.linalg.svd(B)
    # Truncate
    U = U[:, :k]
    s = s[:k]
    V = V[:, :k]
    # Also apply Q1 to U
    U = Q1 @ U

    if returnQ1:
        return U, s, V, Q1
    else:
        return U, s, V


def randomized_nystrom(
    A: np.ndarray,
    l: int,
    Omega1: np.ndarray,
    k: Optional[int] = None,
    return_cond_B: bool = False,
    return_computed_A: bool = True,
    print_cond: bool = False,
):
    n, l_shape = Omega1.shape
    assert (
        A.shape[0] == n and A.shape[1] == n and l == l_shape
    ), "Dimensions do not fit..."

    C = A @ Omega1
    B = Omega1.T @ C
    if print_cond:
        print(f"Testing cond(B): {np.linalg.cond(B)}")
    try:
        L = np.linalg.cholesky(B)  # Hope it works
        if print_cond:
            print(f"Testing cond(L): {np.linalg.cond(L)}")
        Z = np.linalg.solve(L, C.T).T
    except:
        if print_cond:
            print(
                f"Something crashed in the randomized nystrom... Trying LDL!"
            )

        L, D, perm = ldl(B)
        # Force back into L@L.T formulation involves sqrt(D).
        # However, for some semi-definite matricies (where chol fails),
        # D has negative entries... Because of this, we need the absolute value
        L = L @ np.sqrt(np.abs(D))
        # make L upper triangular using the permutation
        L = L[perm, :]
        if print_cond:
            print(f"Testing cond(L): {np.linalg.cond(L)}")
        # Since we permuted rows in L, we need to permute cols in C
        Z = np.linalg.solve(L, C[:, perm].T).T

    Q, R = np.linalg.qr(Z)
    U, s, V = np.linalg.svd(R)
    if not k is None:
        U, s, V = U[:, :k], s[:k], V[:k, :]
    U_hat = Q @ U

    if return_computed_A:
        A_nyst = U_hat @ np.diag(s**2) @ U_hat.T
        if return_cond_B:
            return A_nyst, np.linalg.cond(B)
        else:
            return A_nyst
    elif return_cond_B:
        return U_hat, np.diag(s**2), U_hat.T, np.linalg.cond(B)
    else:
        return U_hat, np.diag(s**2), U_hat.T


def rbf(x, y, c):
    normdiff = np.linalg.norm(x - y, axis=1)
    return np.exp(-(normdiff**2) / c)


# assuming radial basis function
def computeA(X, cval: float):
    n = X.shape[0]
    A = np.zeros((n, n), dtype="d")
    for i in range(n):
        A[i:, i] = rbf(X[i, :], X[i:, :], cval)

    A += np.tril(A, k=-1).T

    return A
