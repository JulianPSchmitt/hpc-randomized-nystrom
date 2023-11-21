import numpy as np
from scipy import linalg
import scipy
from hadamard_transform import hadamard_transform
import torch
np.random.seed(2002)


def rand_nystrom_cholesky(A, Omega, rank):
    """Compute the randomized Nyström rank k approximation given the sketching
    matrix Omega (uses Cholesky decomposition)"""
    C = A@Omega
    B = Omega.T@C
    L = np.linalg.cholesky(B)
    Z = linalg.solve_triangular(L, C.T, lower=True).T
    Q, R = np.linalg.qr(Z)
    U_t, Sigma, V_t = scipy.sparse.linalg.svds(R, k=rank)
    U = Q@U_t
    return U, np.diag(Sigma**2)


def generate_SRHT(l, m):
    """Generate a subsampled randomized Hadamard transform (SRHT) skteching
    matrix"""
    d = np.array([1 if np.random.random() < 0.5 else -1 for i in range(m)])
    omega = np.diag(np.sqrt(m/l)*d)
    omega = hadamard_transform(torch.from_numpy(omega)).numpy()
    return omega[np.random.choice(range(m), l, replace=False), :]
