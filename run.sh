#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Install required dependencies
pip install -r requirements.txt

# Create a directories for the data
mkdir -p ./data/mnist
mkdir -p ./data/YearPredictionMSD

# Download datasets
wget -N -P ./data/mnist https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
wget -N -P ./data/YearPredictionMSD https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2

# Extract datasets
MNIST=./data/mnist/mnist.scale.bz2
if [ -f "$MNIST" ]; then
    echo "Decompressing $MNIST"
    bzip2 -dfk "$MNIST"
fi

YEAR=./data/YearPredictionMSD/YearPredictionMSD.bz2
if [ -f "$YEAR" ]; then
    echo "Decompressing $YEAR"
    bzip2 -dfk "$YEAR"
fi

# Obtain the first 2048 rows of the dataset
head -n 2048 ./data/mnist/mnist.scale > ./data/mnist/mnist_780
head -n 2048 ./data/YearPredictionMSD/YearPredictionMSD > data/YearPredictionMSD/year_780

# Disable BLAS multi-threading to not falsify the results
export OMP_NUM_THREADS=1

# Compute approximation of the MNIST dataset
python code/data.py \
        dataset=MNIST \
        input_dim=2048\
        sketch_dim=400 \
        truncate_rank=200 \
        rbf_smooth=100 \
        variant=CHOLESKY

# Compute approximation of the YearPredictionMSD dataset
python code/data.py \
        dataset=YEAR_PREDICTION_MSD \
        input_dim=2048 \
        sketch_dim=400 \
        truncate_rank=200 \
        rbf_smooth=10000 \
        variant=CHOLESKY

# Compute approximation of the polynomial decay dataset
python code/data.py \
        dataset=POLY_DECAY \
        input_dim=2048 \
        sketch_dim=400 \
        truncate_rank=200 \
        effective_rank=10 \
        data_decay=1 \
        variant=CHOLESKY

# Compute approximation of the exponential decay dataset
python code/data.py \
        dataset=EXP_DECAY \
        input_dim=2048 \
        sketch_dim=400 \
        truncate_rank=200 \
        effective_rank=10 \
        data_decay=0.25 \
        variant=SVD
