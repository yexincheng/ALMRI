#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name=infer_sam_random
#SBATCH --ntasks 1
#SBATCH -c 30
#SBATCH --mem=50000
#SBATCH -o infer_sam_random_out.txt
#SBATCH -e infer_sam_random_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load cuda/10.0.130
module load gnu/5.4.0
# module load anaconda

source activate medsam

srun python infer.py 