export CUDA_VISIBLE_DEVICES=1

# --model_path /shared/eng/pj20/lmr_model/nq_serini_3b/actor/global_step_400 \
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/BM25/squad.py \
    --model_path /shared/eng/pj20/lmr_model/squad_3b_nq_1800/actor/global_step_200 \
    --model_name squad_ours