# Randomized Nyström Low-Rank Approximation

Kernel methods, e.g. support vector machines or Gaussian processes, work with a
high-dimensional or infinite-dimensional feature map and find an optimal
splitting hyperplane. Inner products of data points are stored in a kernel
matrix that serves as a similarity function. The major drawback of these
algorithms is that the computational cost is at least quadratic in the number of
training data points but often becomes cubic due to necessary matrix inversions
or decompositions. To obtain reasonable storage usage and costs for large-scale
problems, low-rank matrix approximations are essential. One of the most popular
examples is the (randomized) Nyström approximation.

## Project Description

In this project, we study the randomized Nyström algorithm for computing a
low-rank approximation of a large and dense matrix $A \in \mathbb{R}^{n \times
n}$ that is symmetric positive semidefinite. Given a sketching matrix $\Omega
\in \mathbb{R}^{n \times l}$, where $l$ is the sketch dimension, the randomized
Nyström approximation relies on the following formula:

$$A_{Nyst} = (A \Omega) (\Omega^T A \Omega)^\dagger (\Omega^T A)$$

where $(\Omega^T A \Omega)^\dagger$ denotes the pseudoinverse of $\Omega^T A
\Omega$. To obtain a rank $k$ approximation $[[A_{Nyst}]]_k$ with $k\leq l$, we
compute a rank $k$ truncation of $A_{Nyst}$. We aim to identify a randomized
algorithm that is numerically stable and scales well on a distributed system. In
particular, the method should be suitable for high-performance computing on a
computer cluster. Our algorithms are implemented in Python 3 and parallelized
via the Message Passing Interface (OpenMPI). For further details, see our
project [report](./report/randomized-nystrom.pdf).

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

To install the required Python packages and dependencies, simply execute the
following from the root directory:

```console
pip install -r requirements.txt
pip install -e .
```

For the synthetic datasets, we provide Python functions that generate the
corresponding matrices. For MNIST and YearPredictionMSD, one has to download the
datasets and map the first $n$ rows into a separate file where $n$ must be a
power of two. (This can always be achieved by zero-padding.) The necessary
commands are shown below for the MNIST data set and $n=2048$ but are similar for
YearPredictionMSD.

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

| Option | Values| Description |
|----------|-------|-------------|
| `dataset` | {MNIST,<br/> YEAR_PREDICTION_MSD,<br/> EXP_DECAY,<br/> POLY_DECAY} | Dataset. |
| `input_dim` | `int` | Dimension $n$ (or number of rows) of the `dataset`.  |
| `sketch_dim` | `int` | Desired dimension $l$ of `sketch` operator. |
| `truncate_rank` | `int` | Rank $k$ after which $A_{Nystr}$ will be truncated. |
| `rbf_smooth` | `int` | Smoothing parameter $c$ for RBF kernel. Only used for `MNIST` and `YEAR_PREDICTION_MSD` data. |
| `effective_rank` | `int` | Effective rank parameter for `EXP_DECAY` and `POLY_DECAY` data. |
| `data_decay` | `float` | Decay parameter for `EXP_DECAY` and `POLY_DECAY` data. |
| `variant` | {CHOLESKY, SVD} | Algorithm to compute the pseudo-inverse $(\Omega^T A \Omega)^\dagger$. |
| `sketch` | {SRHT, SASO} | Sketching operator. Subsampled Randomized Hadamard Transform or Short Axis Sparse Operator.|

For example, one could run:

```console
python code/data.py \
        dataset=MNIST \
        input_dim=2048\
        sketch_dim=400 \
        truncate_rank=200 \
        rbf_smooth=100 \
        variant=CHOLESKY
```

Further examples and code to automatically download the datasets is provided in
[run.sh](./run.sh).

**Note:** We tested the code only in a Linux environment (Fedora 39, Linux
Kernel 6.7.4). Furthermore, the MPI code is specifically designed for 4
processors.

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
