python src/eval/SQL/spider.py \
    --model_path /home/azureuser/cloudfiles/code/DeepRetrieval/training_outputs/spider_3b/actor/global_step_250 \
    --data_path data/sql/spider/val.parquet \
    --model_name spider-3b-val-step-250 \
    --save_dir ../results \
    --batch_size 8