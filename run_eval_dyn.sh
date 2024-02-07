#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --job-name=infer_dyn_e100_random2020
#SBATCH --ntasks 1
#SBATCH -c 30
#SBATCH --mem=50000
#SBATCH -o results/logs/infer_dyn_e100_random2020_out.txt
#SBATCH -e results/logs/infer_dyn_e100_random2020_error.txt
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:a100:1
#SBTACH --account=a_barth

module load cuda
# module load gnu/5.4.0
# module load anaconda

# export WANDB_BASE_URL=http://172.17.0.1:8080

source activate medsam
# wandb login 4aaa2e71cdec13a78a42c6ceac38dd0c7235a131
# wandb offline
srun python eval_dynunet.py --size 192 192 96 \
    --task MRI_LeftKidney \
    --base_model DynUNet_nocache\
    --label_id 1\
    --num_epochs 100 \
    --seed 2020