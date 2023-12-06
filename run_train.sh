#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name=sam_random
#SBATCH --ntasks 1
#SBATCH -c 30
#SBATCH --mem=50000
#SBATCH -o sam_random_out.txt
#SBATCH -e sam_random_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load cuda/10.0.130
module load gnu/5.4.0
# module load anaconda
# export WANDB_BASE_URL=http://172.17.0.1:8080
# export WANDB_API_KEY=4aaa2e71cdec13a78a42c6ceac38dd0c7235a131

source activate medsam
# wandb login 4aaa2e71cdec13a78a42c6ceac38dd0c7235a131
# wandb login --relogin
srun python train.py --task MRI_LeftKidney \
    --label_id 1 \
    --num_epochs 100