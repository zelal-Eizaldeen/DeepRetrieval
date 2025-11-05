#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=run_vllm
#SBATCH --output=logs/run_vllm_%j.out
#SBATCH --error=logs/run_vllm_%j.err
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

echo "=========================================="
echo " Starting VLLM: $(date)"
echo " Running on host: $(hostname)"
echo " SLURM Job ID: $SLURM_JOB_ID"
echo " GPUs requested: $SLURM_GPUS"
echo "=========================================="

# --- Load CUDA module ---
module load cuda/12.6

# --- Activate conda environment ---
source ~/miniconda/etc/profile.d/conda.sh  # adjust if different
conda activate zero2

cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/testDeepRetrieval/DeepRetrieval/

# --- Diagnostic Info ---
echo "=== GPU Info (nvidia-smi) ==="
nvidia-smi
echo "============================="

echo "=== CUDA Toolkit Info (nvcc) ==="
nvcc --version || echo "nvcc not found"
echo "==============================="

echo "=== PyTorch CUDA Info ==="
python - <<'PYCODE'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Detected", torch.cuda.device_count(), "GPU(s)")
    for i in range(torch.cuda.device_count()):
        print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")
        print("   Compute capability:", torch.cuda.get_device_capability(i))
        print("   Memory (GB):", round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2))
print("PyTorch CUDA build version:", torch.version.cuda)
PYCODE
echo "==============================="

# --- Run your actual script ---
echo "Starting main script: "
python query_rewrite.py --query "Who built DeepRetrieval in 2025?"
echo "Finished main script."

echo "=========================================="
echo " Finished  at: $(date)"
echo "=========================================="
