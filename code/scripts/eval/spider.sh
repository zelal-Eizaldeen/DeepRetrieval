export CUDA_VISIBLE_DEVICES=3

# MODEL_PATH=/dev/v-langcao/sft_training_outputs/spider/checkpoint-3135
# MODEL_PATH=/dev/v-langcao/sft_training_outputs/spider/checkpoint-4180
MODEL_PATH=/dev/v-langcao/sft_training_outputs/spider/checkpoint-5225
# MODEL_PATH=/dev/v-langcao/sft_training_outputs/spider/checkpoint-6270
# MODEL_PATH=Qwen/Qwen2.5-3B-Instruct

python src/eval/SQL/spider.py \
    --model_path $MODEL_PATH \
    --data_path data/sql/spider/val.parquet \
    --model_name spider-3b-sft-full \
    --save_dir ../results \
    --with_reasoning True \
    --batch_size 8