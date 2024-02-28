#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --job-name=embedding_all_coronaltest
#SBATCH --ntasks 1
#SBATCH -c 50
#SBATCH --mem=50000
#SBATCH -o results/logs/embedding_allcoronaltest_out.txt
#SBATCH -e results/logs/embedding_allcoronaltest_error.txt
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:a100:1
#SBTACH --account=a_barth
#SBATCH --time=34:00:00

module load cuda
# module load gnu/5.4.0

source activate medsam
srun python pre_embeddings.py --task medsam_MRI_LeftKidney_sagittal \
    --label_id 1 \
    --mode train \
    --filter 3 \
    --view 2 \
    --prefix /scratch/project/bollmann_lab/xincheng/segmentation/ALMRI/datasets/RAINE_organ_51  \
    --checkpoint  checkpoints/MedSAM/medsam_vit_b.pth