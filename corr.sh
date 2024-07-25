#!/bin/bash
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=144:0:0
#SBATCH --mail-user=jyang297@uottawa.ca
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a100:1
module purge
# Change python version if necessary
module load StdEnv/2023
module load python/3.11
module load opencv/4.9

# Source the virtual env

source ~/projects/def-jyzhao/jyang297/torch/bin/activate

cd ~/projects/def-jyzhao/jyang297/correlation_pyramid/Correlation_pyramid/ && torchrun train.py
