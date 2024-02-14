#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name=embedding
#SBATCH --ntasks 1
#SBATCH -c 30
#SBATCH --mem=50000
#SBATCH -o results/logs/embedding_out.txt
#SBATCH -e results/logs/embedding_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load cuda/10.0.130
module load gnu/5.4.0

source activate medsam
srun python pre_embeddings.py --task MRI_LeftKidney_all \
    --label_id 1
    --filter 0