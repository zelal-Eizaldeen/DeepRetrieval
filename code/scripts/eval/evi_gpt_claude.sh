export JAVA_HOME=/Users/patrickjiang/.jdk/jdk-11.0.26+4/Contents/Home

python src/eval/evidence_gpt_claude.py \
    --data_path data/local_index_search/nq_serini/test.parquet \
    --model_name nq-gpt \
    --batch_size 8