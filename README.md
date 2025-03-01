# DeepRetrieval - Powerful Query Generation for Information Retrieval with LLM and RL


![alt text](/images/framework.png "reward curve during training (on pubmed)")



[Wandb Training Report (w/ PubMed Search Engine)](https://api.wandb.ai/links/patjj/zmimgfuq)


**Note: This codebase is built upon the [verl](https://github.com/volcengine/verl) framework**
## Installation

```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## Get started

**Data Preparation**
```
conda activate zero
python examples/data_preprocess/literature_mining.py
```

**Reward function**
```
code/verl/utils/reward_score/literature.py
```
Reward Design:


| Recall      | ≥ 0.7 | ≥ 0.5 | ≥ 0.4 | ≥ 0.3 | ≥ 0.1 | ≥ 0.05 | < 0.05 |
|-------------|-------|-------|-------|-------|-------|--------|--------|
| **Reward**  | +5.0  | +4.0  | +3.0  | +1.0  | +0.5  | +0.1   | -3.5   |

*Table: Reward tiers based on recall performance. Higher recall values receive significantly larger rewards, incentivizing the model to generate more effective queries.*

**Modify the compute_score_fn in code/verl/trainer/main_ppo.py**

**Monitor info**

modify `compute_reward_metrics()` in `code/verl/trainer/ppo/ray_trainer.py`


## Run Training
```
conda activate zero
```

For the following code, if you see Out-of-vram, try add `critic.model.enable_gradient_checkpointing=True` to the script


**Run 3B model (Qwen/Qwen2.5-3B-Instruct):**
```
sh code/scripts/literature_search_train.sh 
```

### Reward Curve During Training

![alt text](/images/reward_curve.png "reward curve during training (on pubmed)")


## Run Evaluation

```
sh code/scripts/eval/inst/liter.sh
```

**Result**

| Model | Method | Recall (Publication) | Recall (Trial) |
|-------|--------|----------------------|----------------|
| **GPT-4o** | Zero-shot | 5.79 | 6.74 |
| | Few-shot | 7.67 | 4.69 |
| | ICL | 19.72 | 14.26 |
| | ICL+Few-shot | 11.95 | 7.98 |
| **GPT-3.5** | Zero-shot | 4.01 | 3.37 |
| | Few-shot | 4.15 | 3.34 |
| | ICL | 18.68 | 13.94 |
| | ICL+Few-shot | 7.06 | 5.54 |
| **Haiku-3** | Zero-shot | 10.98 | 11.59 |
| | Few-shot | 14.71 | 7.47 |
| | ICL | 20.92 | 24.68 |
| | ICL+Few-shot | 19.11 | 9.27 |
| **Mistral-7B** | Zero-shot | 7.18 | 8.08 |
| **LEADS** | Zero-shot | 24.68 | 32.11 |
| **DeepRetrieval** | Zero-shot | **60.82** | **70.84** |

*Table: Comparison of different models and methods on publication search and trial search tasks. Bold numbers indicate the best performance.*


---
Thanks for your interests!