#!/bin/bash --login
#SBATCH --nodes 1
#SBATCH --job-name=DynUNet_e100random2024
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=50
#SBATCH --mem=50G
#SBATCH -o results/logs/DynUNet_e100random2024_out.txt
#SBATCH -e results/logs/DynUNet_e100random2024_error.txt
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:a100:1
#SBTACH --account=a_barth
#SBATCH --time=34:00:00

module load cuda
# module load gnu/5.4.0
# module load anaconda

# export TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0;7.5;8.0;8.6;9.0"

source activate medsam

srun python active_dynunet.py --size 192 192 96 \
    --num_epochs 100 \
    --task MRI_LeftKidney \
    --base_model DynUNet\
    --seed 2024 