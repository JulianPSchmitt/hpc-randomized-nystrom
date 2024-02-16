# Randomized Nyström Low-Rank Approximation

## Project Description

In this project, we study the randomized Nyström algorithm for computing a
low-rank approximation of a large and dense matrix $A \in \mathbb{R}^{n \times
n}$ that is symmetric positive semidefinite. Given a sketching matrix $\Omega
\in \mathbb{R}^{n \times l}$, where $l$ is the sketch dimension, the randomized
Nyström approximation relies on the following formula:

$$A_{Nyst} = (A \Omega) (\Omega^T A \Omega)^\dagger (\Omega^T A)$$

where $(\Omega^T A \Omega)^\dagger$ denotes the pseudoinverse of $\Omega^T A
\Omega$. To obtain a rank-$k$ approximation $[[A_{Nyst}]]_k$ with $k\leq l$, we
compute a rank-$k$ truncation of $A_{Nyst}$. We aim to identify a randomized
algorithm that is numerically stable and scales well on a distributed system.
Our algorithms are implemented in Python 3 and parallelized via the Message
Passing Interface (OpenMPI). For further details, see our project
[report](./report/Project2.pdf).

## Datasets

We use two synthetic and two real-world datasets to evaluate our algorithms. The
synthetic data is described in Section 5 of [[3]](#3). In particular, we worked
with the so-called *polynomial decay* and *exponential decay* matrices.

For the real-world data, we considered the MNIST [[2]](#1) and YearPredictionMSD
[[1]](#1) datasets. The radial basis function
$\exp\left(-\frac{||x_i-x_j||^2}{c^2}\right)$ is used to build a dense matrix $A
\in \mathbb{R}^{n \times n}$ from $n$ rows of the input data. Regarding the
parameter $c$, we set $c=100$ for MNIST and $c=10^4$ as well as $c=10^5$ for
YearPredictionMSD.

## How to run the Algorithms

### Setup Datasets and Python Code

To obtain the Nyström approximation for the first $n$ rows of the datasets
mentioned above, one can follow the steps below. The steps are shown using the
MNIST data set as an example but are similar for the other datasets.

1. (Optional) Using Python 3.12.1, one can install the required
   packages/dependencies from the project's root directory by executing:

    ```console
    pip install -r requirements.txt
    ```

2. The MNIST dataset can be obtained by the commands below:

    ```console
    # Create a directory for the data
    mkdir -p data/mnist
    cd data/mnist

    # Download and extract the dataset
    wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
    bzip2 -d mnist.scale.bz2

    # Obtain the first 2048 rows of the dataset
    head -n 2048 mnist.scale > mnist_780
    ```

3. To compute the Nyström approximation for the MNIST dataset sequentially on a
   single processor, run the following with desired arguments:

    ```console
    python code/data.py dataset=MNIST
    ```

All steps for all 4 datasets are implemented in [run.sh](./run.sh). Moreover,
the script computes the Nyström approximations and outputs the corresponding
approximation error.

**Note:** We tested the code only in a Linux environment (Fedora 39, Linux
Kernel 6.7.4). Furthermore, the MPI code is specifically designed for 4
processors.

### Usage of the Python CLI

Our Python implementations can be accessed via the file `code/data.py`.
Arguments are passed using the command line.

For sequential execution run:

```console
python [Options...]
```

For parallel execution on 4 processors run:

```console
mpiexec -n 4 python [Options...]
```

| Argument | Values| Description |
|----------|-------|-------------|
| `dataset` | {MNIST,<br/> YEAR_PREDICTION_MSD,<br/> EXP_DECAY,<br/> POLY_DECAY} | Dataset. |
| `input_dim` | `int` | Dimension $n$ (or number of rows) of the `dataset`.  |
| `sketch_dim` | `int` | Desired dimension $l$ of `sketch` operator. |
| `truncate_rank` | `int` | Rank $k$ after which $A_{Nystr}$ will be truncated. |
| `rbf_smooth` | `int` | Smoothing parameter $c$ for RBF kernel.<br/> Only used for MNIST and YEAR_PREDICTION_MSD data. |
| `effective_rank` | `int` | Effective rank parameter for EXP_DECAY and POLY_DECAY data. |
| `data_decay` | `float` | Decay parameter for EXP_DECAY and POLY_DECAY data. |
| `variant` | {CHOLESKY, SVD} | Algorithm to compute the pseudo-inverse $(\Omega^T A \Omega)^\dagger$. |
| `sketch` | {SRHT, SASO} | Sketching operator. Subsampled Randomized Hadamard Transform<br/> or Short Axis Sparse Operator.|

## References

<a id="1">[1]</a>
T. Bertin-Mahieux, D. P. Ellis, B. Whitman, and P. Lamere. The million song
dataset. In Proceedings of the 12th International Conference on Music
Information Retrieval (ISMIR 2011), 2011.

<a id="2">[2]</a>
Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied
to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

<a id="3">[3]</a>
J. A. Tropp et al. Fixed-Rank Approximation of a
Positive-Semidefinite Matrix from Streaming Data. 2017. arXiv: 1706.05736
[cs.NA]. 14
