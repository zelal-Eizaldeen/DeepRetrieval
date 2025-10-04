#!/bin/bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=jimeng-ic
#SBATCH --job-name=eval_pubmed_32
#SBATCH --output=logs/eval_pubmed_32_%j.out
#SBATCH --error=logs/eval_pubmed_32_%j.err
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

echo "Starting Eval pubmed at: $(date)" 

module load cuda/12.6

conda activate zero
cd /projects/illinois/eng/cs/jimeng/zelalae2/scratch/DeepRetrieval/code/


sh scripts/eval/pubmed_32.sh




echo "Finished Eval pubmed_sh at: $(date)"