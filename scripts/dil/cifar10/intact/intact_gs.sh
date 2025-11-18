#!/bin/bash
#SBATCH --job-name=split_cifar10_resnet18_intact_dil
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/dil/cifar10/intact/intact_gs"