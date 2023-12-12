import numpy as np
from os.path import join
from __init__ import _FOLDER


def pol_decay(n, r, p):
    A = [1.0 for _ in range(r)] + [(2.0 + o) ** (-p) for o in range(n - r)]
    A = np.diag(A)

    return A


def exp_decay(n, r, q):
    A = [1.0 for _ in range(r)] + [(10.0) ** (-(o + 1)) for o in range(n - r)]
    A = np.diag(A)

    return A


if __name__ == "__main__":
    n = 2**13
    r = 10
    p = 1.0
    q = 0.25

    A_PD = pol_decay(n, r, p)
    A_ED = exp_decay(n, r, q)

    test_matricies = np.array([A_PD, A_ED])
    print(test_matricies.shape)

    np.save(join(_FOLDER, "data", "test_matricies_8192.npy"), test_matricies)
