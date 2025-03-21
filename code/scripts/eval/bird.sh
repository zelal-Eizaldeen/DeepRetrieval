export CUDA_VISIBLE_DEVICES=3


# MODEL_PATH=/dev/v-langcao/sft_training_outputs/bird_wo_reasoning/checkpoint-6342
# MODEL_PATH=/dev/v-langcao/sft_training_outputs/bird_wo_reasoning/checkpoint-8456
# MODEL_PATH=/dev/v-langcao/sft_training_outputs/bird_wo_reasoning/checkpoint-10570
MODEL_PATH=/dev/v-langcao/sft_training_outputs/bird_wo_reasoning/checkpoint-12684
# MODEL_PATH=Qwen/Qwen2.5-3B-Instruct

python src/eval/SQL/bird.py \
    --model_path $MODEL_PATH \
    --data_path data/sql/bird/test.parquet \
    --model_name bird-3b-sft-full \
    --save_dir ../results \
    --with_reasoning False \
    --batch_size 8