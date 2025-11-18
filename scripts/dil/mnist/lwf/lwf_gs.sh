#!/bin/bash
#SBATCH --job-name=split_mnist_mlp_lwf_dil
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/dil/mnist/lwf/lwf_gs"
