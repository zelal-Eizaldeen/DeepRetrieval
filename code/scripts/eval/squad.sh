export CUDA_VISIBLE_DEVICES=1

# --model_path /shared/eng/pj20/lmr_model/squad_3b_nq_1800/actor/global_step_200 \
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/BM25/squad.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --model_name squad_base