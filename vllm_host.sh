export CUDA_VISIBLE_DEVICES=0,1

python3 -m vllm.entrypoints.openai.api_server \
    --model /shared/rsaas/pj20/lmr_model/nq_serini_3b_continue/actor/global_step_1400 \
    --port 8000 \
    --max-model-len 2048 \
    --tensor-parallel-size 2 


# --model DeepRetrieval/DeepRetrieval-NQ-BM25-3B
# --model DeepRetrieval/DeepRetrieval-TriviaQA-BM25-3B \
# --model DeepRetrieval/DeepRetrieval-SQuAD-BM25-3B-200 \