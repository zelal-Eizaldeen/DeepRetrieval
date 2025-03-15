export CUDA_VISIBLE_DEVICES=6

# --model_path /shared/eng/pj20/lmr_model/nq_serini_3b/actor/global_step_400 \
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/BM25/triviaqa.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --model_name tqa-base-3b