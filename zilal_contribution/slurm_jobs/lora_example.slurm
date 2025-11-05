#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=lora_exmaple
#SBATCH --output=logs/lora_example_%j.out
#SBATCH --error=logs/lora_example_%j.err
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

echo "=========================================="
echo " Starting Cluster infoat: $(date)"
echo " Running on host: $(hostname)"
echo " SLURM Job ID: $SLURM_JOB_ID"
echo " GPUs requested: $SLURM_GPUS"
echo "=========================================="

# --- Load CUDA module ---
module load cuda/12.6

# --- Activate conda environment ---
source /projects/illinois/eng/cs/jimeng/zelalae2/miniconda/etc/profile.d/conda.sh # adjust if different

conda activate zero


cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/DeepRetrieval/code

# --- Run your actual script ---
echo "Starting main script"
sh scripts/train/pubmed_32.sh 

echo "Finishing main script"


