#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name=infer_sam_random150
#SBATCH --ntasks 1
#SBATCH -c 30
#SBATCH --mem=50000
#SBATCH -o infer_sam_random150_out.txt
#SBATCH -e infer_sam_random150_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load cuda/10.0.130
module load gnu/5.4.0
# module load anaconda

# export WANDB_BASE_URL=http://172.17.0.1:8080

source activate medsam
# wandb login 4aaa2e71cdec13a78a42c6ceac38dd0c7235a131
# wandb offline
srun python eval.py --base_model MedSAM --num_epochs 150 