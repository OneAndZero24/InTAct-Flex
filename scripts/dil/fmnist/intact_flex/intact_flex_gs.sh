#!/bin/bash
#SBATCH --job-name=split_fmnist_mlp_intact_flex_dil
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=dgx

source scripts/main.sh

run_sweep_and_agent "scripts/dil/fmnist/intact_flex/intact_flex_gs"