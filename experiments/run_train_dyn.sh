#!/bin/bash --login
#SBATCH --nodes 1
#SBATCH --job-name=seed2024dyn_active_ts45_random
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=50
#SBATCH --mem=50G
#SBATCH -o results/logs/seed2024dyn_active_ts45_random_out.txt
#SBATCH -e results/logs/seed2024dyn_active_ts45_random_error.txt
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=a_ai_collab
#SBATCH --time=72:00:00



source activate medsam

cd "$(dirname "$0")/.."
mkdir -p results/logs

srun python active_dynunet.py active \
    --strategy random \
    --spatial_size 192 192 96 \
    --initial_subjects 5 \
    --initial_epochs 100 \
    --incremental_epochs 30 \
    --seed 2024 \
    --learning_rate 0.0001
