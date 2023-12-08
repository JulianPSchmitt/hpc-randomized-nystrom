import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from randomized_nystrom import rand_nystrom_cholesky

R = [5, 10, 20]
epss = [1e-4, 1e-2, 1e-1]
ps = [0.5, 1.0, 2.0]
qs = [0.1, 0.25, 1.0]
matrix_variables = [epss, ps, qs]


def properties_example_matricies(
    savepath: str,
    test_matrix_path: str,
    l: int = 50,
    plotTitleSpecifier: str = "",
):
    test_matricies = np.load(test_matrix_path)
    # Variables used to generate above matricies
    types = ["Low Rank + noise", "Pol. Decay", "Exp. Decay"]
    var_names = "epq"

    fig_sing, axs_sing = plt.subplots(3, 3, figsize=(15, 10))
    fig_diag, axs_diag = plt.subplots(3, 3, figsize=(15, 10))
    fig_err, axs_err = plt.subplots(1, 3, figsize=(15, 5))
    fig_condA, axs_condA = plt.subplots(1, 3, figsize=(15, 5))
    fig_condB, axs_condB = plt.subplots(1, 3, figsize=(15, 5))

    for k in range(3):  # Over noise/pol/exp matrix
        for i in range(3):  # Over R-values
            errors = []
            cond_As = []
            cond_Bs = []
            for j in range(3):  # Over eps/ps/qs variables
                A = test_matricies[k, i, j]
                ### Plots related to the matricies themselves ###
                # Diagonal entries
                axs_diag[k, i].loglog(
                    np.diag(A),
                    label=f"with {var_names[k]}={matrix_variables[k][j]}",
                )
                # Singular values
                sing_vals = np.linalg.svd(A, compute_uv=False)
                axs_sing[k, i].loglog(
                    sing_vals,
                    label=f"with {var_names[k]}={matrix_variables[k][j]}",
                )

                ### values related to rand_nystrom ###
                try:
                    n, _ = A.shape
                    Omega = np.random.random((n, l))
                    U, S = rand_nystrom_cholesky(A=A, Omega=Omega, rank=l)
                    A_nyst = U @ S @ U.T

                    rel_err = np.linalg.norm(A_nyst - A) / np.linalg.norm(A)
                    errors.append(rel_err)
                    cond_As.append(np.linalg.cond(A))
                    cond_Bs.append(
                        np.linalg.cond(Omega.T @ A @ Omega)
                    )  # cond_B
                except Exception as err:
                    print(
                        f"For matrix {types[k]} with R={R[i]}, error"
                        f" encountered: {err}!"
                    )
                    errors.append(np.inf)
                    cond_As.append(np.linalg.cond(A))
                    cond_Bs.append(np.linalg.cond(Omega.T @ A @ Omega))

            ### Error plots ###
            axs_err[k].loglog(
                matrix_variables[k],
                errors,
                label=f"R={R[i]}",
            )
            axs_err[k].set_title(f"Type: {types[k]}")
            axs_err[k].legend()
            axs_err[k].set_xlabel(f"{var_names[k]} value")
            if k == 0:
                axs_err[k].set_ylabel("$||A_{nyst}-A||_2\;/\;||A||_2$")
            ### Condition plots ###
            print(f"CONDITION_A: {cond_As}")
            axs_condA[k].loglog(
                matrix_variables[k],
                cond_As,
                label=f"R={R[i]}",
            )
            axs_condA[k].set_title(f"Type: {types[k]}")
            axs_condA[k].legend()
            axs_condA[k].set_xlabel(f"{var_names[k]} value")
            if k == 0:
                axs_condA[k].set_ylabel("cond$(A)$")
            print(f"CONDITION_B: {cond_Bs}")
            axs_condB[k].loglog(
                matrix_variables[k],
                cond_Bs,
                label=f"R={R[i]}",
            )
            axs_condB[k].set_title(f"Type: {types[k]}")
            axs_condB[k].legend()
            axs_condB[k].set_xlabel(f"{var_names[k]} value")
            if k == 0:
                axs_condB[k].set_ylabel("cond$(B)$")

            # Adjust titles and legends
            axs_diag[k, i].set_title(f"Type: {types[k]}, R={R[i]}")
            if k != 0:
                axs_diag[k, i].legend(loc="lower left")
            else:
                axs_diag[k, i].legend(loc="upper right")
            axs_sing[k, i].set_title(f"Type: {types[k]}, R={R[i]}")
            axs_sing[k, i].legend(loc="lower left")
            if k == 2:
                axs_diag[k, i].set_xlabel("Index")
                axs_sing[k, i].set_xlabel("Index")
            if i == 0:
                axs_diag[k, i].set_ylabel("Value of diagonal entry")
                axs_sing[k, i].set_ylabel("Singular value")

    # Show plots
    fig_diag.tight_layout()
    fig_sing.tight_layout()
    fig_err.tight_layout()
    fig_condA.tight_layout()
    fig_condB.tight_layout()
    # save figures
    fig_diag.savefig(
        join(
            savepath,
            "ExampleMatriciesDiag" + plotTitleSpecifier + ".png",
        )
    )
    fig_sing.savefig(
        join(
            savepath,
            "ExampleMatriciesSingVals" + plotTitleSpecifier + ".png",
        )
    )
    fig_err.savefig(
        join(
            savepath,
            "ExampleMatriciesRelErrors" + plotTitleSpecifier + ".png",
        )
    )
    fig_condA.savefig(
        join(
            savepath,
            "ExampleMatriciesCondA" + plotTitleSpecifier + ".png",
        )
    )
    fig_condB.savefig(
        join(
            savepath,
            "ExampleMatriciesCondB" + plotTitleSpecifier + ".png",
        )
    )
    plt.show()
