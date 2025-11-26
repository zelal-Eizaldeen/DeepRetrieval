#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=build_docker_image
#SBATCH --output=logs/build_docker_image_%j.out
#SBATCH --error=logs/build_docker_image_%j.err
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00

echo "=========================================="
echo " Starting at: $(date)"
echo " Running on host: $(hostname)"
echo " SLURM Job ID: $SLURM_JOB_ID"
echo " GPUs requested: $SLURM_GPUS"
echo "=========================================="

# --- Load CUDA module ---
module load cuda/12.6
# --- Activate conda environment ---
source /projects/illinois/eng/cs/jimeng/zelalae2/miniconda/etc/profile.d/conda.sh  # adjust if different
conda activate zero

cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/original_deepRetrieval/DeepRetrieval/code/docker

docker buildx build \
  --platform linux/amd64 \
  -t deepretrieval-pubmed \
  -f Dockerfile.ngc.vllm .

docker run --gpus all -p 8000:8000 deepretrieval-pubmed