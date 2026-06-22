#!/bin/bash --login
#SBATCH --nodes 1
#SBATCH --job-name=samroi_e20random2024
#SBATCH --ntasks 1
#SBATCH -c 30
#SBATCH --mem=50000
#SBATCH -o results/logs/samroi_e20random2024_out.txt
#SBATCH -e results/logs/samroi_e20random2024_error.txt
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:1
#SBATCH --account=a_barth
#SBATCH --time=34:00:00

module load cuda/11.7.0
# module load gnu/5.4.0
# module load anaconda

source activate medsam

srun python train.py --task MRI_LeftKidney \
    --label_id 1 \
    --base_model SAM_roi \
    --checkpoint ./checkpoints/SAM/sam_vit_b_01ec64.pth \
    --num_epochs 20 \
    --seed 2024