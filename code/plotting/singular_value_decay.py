import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from mnist import build_A
from tqdm import tqdm
from __init__ import _FOLDER


if __name__ == "__main__":

    def rbf_kernel(X, n, c):
        X = X[:n, :]
        print(f"Shape: {X.shape}")
        A = build_A(X, c=c, save=False)
        return A

    n = 2**13
    all_As = []
    dataset_names = []

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
    print(f"Going to pol with shape: {A_pol.shape}")
    all_As.append(A_pol)
    dataset_names.append("Pol-R10-p1")
    A_exp = test_matricies[1]
    all_As.append(A_exp)
    dataset_names.append("Exp-R10-q0.25")

    # From here we generate plots for all A and all methods
    for i, A in tqdm(enumerate(all_As), total=len(all_As)):
        if dataset_names[i] in ["Pol-R10-p1", "Exp-R10-q0.25"]:
            sing_vals = np.diag(A)
        else:
            sing_vals = np.linalg.svd(A, compute_uv=False)
        fig, ax = plt.subplots(1, 1)
        ax.loglog(sing_vals, "ko-")
        ax.set_title(f"Singular values of {dataset_names[i]}")
        ax.set_xlabel("i")
        ax.set_ylabel("$\sigma_i(A)$")
        plt.tight_layout()
        plt.savefig(
            join(
                _FOLDER,
                "plots",
                f"singular_values_{dataset_names[i]}.png",
            )
        )
        plt.show(block=False)
