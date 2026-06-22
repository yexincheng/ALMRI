#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name=infer_sam_e20_random2024
#SBATCH --ntasks 1
#SBATCH -c 30
#SBATCH --mem=50000
#SBATCH -o results/logs/infer_sam_e20_random2024_out.txt
#SBATCH -e results/logs/infer_sam_e20_random2024_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load cuda/10.0.130
module load gnu/5.4.0
# module load anaconda


source activate medsam

srun python eval.py --task MRI_LeftKidney \
    --label_id 1 \
    --num_epochs 20 \
    --seed 2024