#!/bin/bash
#SBATCH --job-name=cifar10_resnet18_si_optimal_hyperparams
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/optimal_hyperparams/cil/cifar10/si/si"
