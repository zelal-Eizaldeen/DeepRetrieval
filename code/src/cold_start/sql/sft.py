import random
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)



dataset = 'bird'
# dataset = 'spider'

# for_cold_start = True
cold_start_data_size = 1000
for_cold_start = False


model_name = 'Qwen/Qwen2.5-3B-Instruct'



################
# Dataset
################

train_data = []
with open(f'data/sql/cold_start/{dataset}_reason_sft_train.jsonl', 'r') as f:
    for line in f:
        D = json.loads(line)
        if D != {}:
            train_data.append(D['new_prompt'])

random.seed(42)
if for_cold_start:
    train_data = random.sample(train_data, cold_start_data_size)


train_dataset = Dataset.from_dict({
    "source": [f'{dataset}_reason_sft_train' for _ in range(len(train_data))],
    "messages": train_data,
})


################
# Model
################

model_kwargs = dict(
    trust_remote_code=True,
    attn_implementation='flash_attention_2',
    torch_dtype=torch.bfloat16,
    use_cache=False,
    device_map=get_kbit_device_map(),
)
 
model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token




################
# Training
################
training_args = SFTConfig(
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    save_strategy='epoch',
    output_dir="/shared/eng/pj20/lmr_model/cold_start",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    processing_class=tokenizer,
)

trainer.train()