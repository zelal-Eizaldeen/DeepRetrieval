python src/eval/SQL/bird.py \
    --model_path /dev/v-langcao/training_outputs/bird_3b/actor/global_step_450 \
    --data_path data/sql/bird/test.parquet \
    --model_name bird-3b-step-450 \
    --save_dir ../results \
    --batch_size 8