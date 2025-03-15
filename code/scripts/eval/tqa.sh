export CUDA_VISIBLE_DEVICES=3

# --model_path /shared/eng/pj20/lmr_model/nq_serini_3b/actor/global_step_400 \
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/BM25/triviaqa.py \
    --model_path /shared/eng/pj20/lmr_model/nq_serini_3b/actor/global_step_800 \
    --model_name tqa-3b-step800