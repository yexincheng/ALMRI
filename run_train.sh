#!/bin/bash --login
#SBATCH --nodes 1
#SBATCH --job-name=sam_e150random2024
#SBATCH --ntasks 1
#SBATCH -c 30
#SBATCH --mem=50000
#SBATCH -o results/logs/sam_e150random2024_out.txt
#SBATCH -e results/logs/sam_e150random2024_error.txt
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#sbatch --account=a_barth

module load cuda/11.7.0
# module load gnu/5.4.0
# module load anaconda
# export WANDB_BASE_URL=http://172.17.0.1:8080
# export WANDB_API_KEY=4aaa2e71cdec13a78a42c6ceac38dd0c7235a131

source activate medsam
# wandb login 4aaa2e71cdec13a78a42c6ceac38dd0c7235a131
# wandb login --relogin
srun python train.py --task MRI_LeftKidney \
    --label_id 1 \
    --base_model SAM \
    --checkpoint ./checkpoints/SAM/sam_vit_b_01ec64.pth \
    --num_epochs 150 \
    --seed 2024