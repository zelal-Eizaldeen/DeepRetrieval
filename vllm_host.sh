export CUDA_VISIBLE_DEVICES="1"
# echo $CUDA_VISIBLE_DEVICES
python3 -m vllm.entrypoints.openai.api_server \
    --model DeepRetrieval/DeepRetrieval-PubMed-3B-Llama \
    --port 8000 \
    --max-model-len 2048 \
    --tensor-parallel-size 1 \


# --model DeepRetrieval/DeepRetrieval-NQ-BM25-3B
# --model DeepRetrieval/DeepRetrieval-TriviaQA-BM25-3B \
# --model DeepRetrieval/DeepRetrieval-SQuAD-BM25-3B-200 \