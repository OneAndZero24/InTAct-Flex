#!/bin/bash
#SBATCH --job-name=cifar10_dil_resnet18_lwf_optimal_hyperparams
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/optimal_hyperparams/dil/cifar10/lwf/lwf"
