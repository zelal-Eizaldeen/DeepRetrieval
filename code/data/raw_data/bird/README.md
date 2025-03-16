# BIRD Raw Data Process

References:
* Data: https://bird-bench.github.io/
* Code: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird


```bash
cd data/raw_data/bird

wget https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip

unzip train.zip -d .
unzip dev.zip -d .

mv dev_20240627 dev

unzip train/train_databases -d train
unzip dev/dev_databases -d dev
```