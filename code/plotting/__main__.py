import csv
import numpy as np
from plotting.srhtplots import srht_plot
from plotting.randomized_svd import singvalplots
from plotting.example_matricies import properties_example_matricies
from plotting.papermatchingplots import paperMatchingPlots
from os.path import join
from __init__ import _FOLDER


def getWb(path: str, simple: bool = False):
    W = []
    b = []
    with open(path, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")  # , quotechar='|'
        for i, row in enumerate(spamreader):
            if i != 0:
                if simple:
                    W.append([float(val) for val in row[:-1]])
                    b.append(float(row[-1]))
                else:
                    row = row[1:] + row[0:1]
                    W.append([float(val) for val in row[2:]])
                    b.append(float(row[1]))
    return np.array(W), np.array(b)


def load_simple_data(simple: bool = False):
    path = join(_FOLDER, "data")

    if simple:
        path = join(path, "Experience-Salary.csv")
    else:
        path = join(path, "Walmart.csv")

    W, b = getWb(path=path, simple=simple)

    W = np.array(W)
    b = np.array(b)

    # make sure rows is power of 2
    numrows = W.shape[0]
    newrows = 2 ** (int(np.log2(numrows)))
    if newrows != numrows:
        print(
            "WARNING! Number of rows removed to get power of 2. Was"
            f" {numrows} rows, now is {newrows}"
        )
    W = W[:newrows]
    b = b[:newrows]

    return W, b


if __name__ == "__main__":
    plotpath = join(_FOLDER, "plots")
    specifier = "_tmp"

    # week 7, ex1
    # Properties (and time to run) for SRHT (single processor)
    W, b = load_simple_data(simple=False)
    b = b[:, None]
    srht_plot(W, b, savepath=plotpath, plotTitleSpecifier=specifier)
    # Randomized SVD and its properties (singular values etc.)
    # week 7, ex2
    singvalplots(savepath=plotpath, plotTitleSpecifier=specifier)

    # week 8, ex2
    # Example matricies (noise, pol and exp) and their
    # properties (singular values etc.)
    test_matricies_path = join(
        _FOLDER,
        "data",
        "test_matricies.npy",
    )
    properties_example_matricies(
        savepath=plotpath,
        test_matrix_path=test_matricies_path,
        plotTitleSpecifier=specifier,
    )

    # Week 9
    # Plots matching what we see in the paper in terms of (relative)
    # precision of two datasets MNIST and YearMSD.
    # Note: Both MNIST and YearMSD cannot work with randomized nystrom chol.
    pathX1 = join(_FOLDER, "data", "MNISTtrain.npy")
    pathX2 = join(_FOLDER, "data", "YearMSD.npy")
    paperMatchingPlots(
        savepath=plotpath,
        MNIST_X_path=pathX1,
        YearMSD_X_path=pathX2,
        n=2**8,
        plotTitleSpecifier=specifier,
    )
