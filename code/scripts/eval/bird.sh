export CUDA_VISIBLE_DEVICES=1


# MODEL_PATH=/dev/v-langcao/sft_training_outputs/bird_wo_reasoning/checkpoint-6342
# MODEL_PATH=/dev/v-langcao/sft_training_outputs/bird_wo_reasoning/checkpoint-8456
# MODEL_PATH=/dev/v-langcao/sft_training_outputs/bird_wo_reasoning/checkpoint-10570
# MODEL_PATH=/dev/v-langcao/DeepRetrieval-SQL/cold_start/bird_Qwen/Qwen2.5-Coder-3B-Instruct/checkpoint-8456
# MODEL_PATH=/dev/v-langcao/DeepRetrieval-SQL/cold_start/bird_Qwen/Qwen2.5-Coder-3B-Instruct_wo_reasoning/checkpoint-4228/
# MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
# MODEL_PATH=Qwen/Qwen2.5-Coder-3B-Instruct
MODEL_PATH=/dev/v-langcao/qwen-7

python src/eval/SQL/bird.py \
    --model_path $MODEL_PATH \
    --data_path data/sql/bird/test.parquet \
    --model_name bird-3b-sft-full \
    --save_dir ../results \
    --with_reasoning true \
    --batch_size 8