#!/bin/bash --login
#SBATCH --nodes 1
#SBATCH --job-name=infer_dyn_trainingset51
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=30
#SBATCH --mem=50G
#SBATCH -o results/logs/infer_dyn_trainingset51_out.txt
#SBATCH -e results/logs/infer_dyn_trainingset51_error.txt
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=a_ai_collab
#SBATCH --time=72:00:00

module load cuda

source activate medsam

cd "$(dirname "$0")/.."
mkdir -p results/logs

srun python eval_dynunet.py infer \
    --spatial_size 192 192 96 \
    --checkpoint checkpoints/DynUNet_trainingset45_entropy_active/seed2025/init100_inc30/DynUNet_45subjects_seed2025.pt \
    --image_dir datasets/trainingset_51 \
    --label_dir datasets/trainingset_51/labels/final \
    --pred_dir inference_outputs/pred_trainingset_51 \
    --dice_dir results/dice/trainingset_51 \
    --save_predictions true
