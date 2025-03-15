python src/eval/SQL/bird.py \
    --model_path /home/azureuser/cloudfiles/code/DeepRetrieval/training_outputs/bird_3b/actor/global_step_100 \
    --data_path data/sql/bird/test.parquet \
    --model_name bird-3b-step-100 \
    --save_dir ../results \
    --batch_size 8