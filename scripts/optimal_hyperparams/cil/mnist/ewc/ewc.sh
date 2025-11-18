#!/bin/bash
#SBATCH --job-name=split_mnist_mlp_ewc_cil
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/optimal_hyperparams/cil/mnist/ewc/ewc"
