import csv
import numpy as np
from plotting.srhtplots import srht_plot
from plotting.singvals import singvalplots
from plotting.example_matricies import properties_example_matricies
from plotting.papermatchingplots import paperMatchingPlots
from os.path import join


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


def load_data(simple: bool = False):
    path = "/home/chrillebon/hpc_epfl/Project2andFriends/week7/"
    if simple:
        path += "Experience-Salary.csv"
    else:
        path += "Walmart.csv"

    try:
        W, b = getWb(path=path, simple=simple)
    except:
        path = "Project2andFriends/" + path
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
    plotpath = "/home/chrillebon/hpc_epfl/Project2andFriends/hpc-randomized-nystrom/plots"
    specifier = "_tmp"

    # week 7, ex1
    W, b = load_data(simple=False)
    b = b[:, None]
    srht_plot(W, b, savepath=plotpath, plotTitleSpecifier=specifier)
    # week 7, ex2
    singvalplots(savepath=plotpath, plotTitleSpecifier=specifier)

    # week 8, ex2
    test_matricies_path = join(
        "/home/chrillebon/hpc_epfl/Project2andFriends/week8",
        "test_matricies.npy",
    )
    properties_example_matricies(
        savepath=plotpath,
        test_matrix_path=test_matricies_path,
        plotTitleSpecifier=specifier,
    )

    # Week 9 (datasets and paper)
    pathX1 = join(
        "/home/chrillebon/hpc_epfl/Project2andFriends/week9", "MNISTtrain.npy"
    )
    pathX2 = join(
        "/home/chrillebon/hpc_epfl/Project2andFriends/week9", "YearMSD.npy"
    )
    paperMatchingPlots(
        savepath=plotpath,
        MNIST_X_path=pathX1,
        YearMSD_X_path=pathX2,
        n=2**8,
        plotTitleSpecifier=specifier,
    )