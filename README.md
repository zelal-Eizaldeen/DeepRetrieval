<div align="center">

# DeepRetrieval - Hacking Search Engines & Retrievers with LLM via RL  
### **Let LLMs learn how to search!**

<p align="center">
  <a href="https://arxiv.org/abs/2503.00223">
    <img src="https://img.shields.io/badge/arXiv-2503.00223-b31b1b.svg" alt="ArXiv">
  </a>
  <a href="https://huggingface.co/DeepRetrieval">
    <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" height="28">
  </a>
</p>

</div>





![alt text](/images/framework.png "reward curve during training (on pubmed)")

![alt text](/images/performance_overview.png "performance overview")

[Preliminary Technical Report (ArXiv preprint)](https://arxiv.org/pdf/2503.00223)

[Example Wandb Training Log on PubMed Search Engine](https://wandb.ai/patjj/literature_search?nw=nwuserpj20)


## 📦 Installation

**General Installation (for all retrieval methods):**
```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
cd code
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib huggingface_hub
```

**For Search Engine Retrieval, you can skip the following steps.**

If using sparse retrieval (e.g., BM25) or dense retrieval (e.g., DPR), please also install the following:
```
# we use pyserini for efficient retrieval and evaluation
pip install pyserini    # the version we used is 0.22.1

# if you don't have faiss installed, install it with:
pip install faiss-gpu==1.7.2    # the version we used is 1.7.2

# if you don't have java installed, install it with:
pip install install-jdk && python -c "import jdk; jdk.install('11')"

# support sql execution
pip install func_timeout
```

## 🫧 Get Started

### **1. Dataset Download/Preprocess**

We provide two options for data preparation:

<details>
<summary style="font-weight: bold;">🚀 Option 1: Download pre-processed datasets from Huggingface (Recommended)</summary>

All preprocessed datasets are available on our Huggingface repository. You can download them using the provided script:

```bash
cd code
# List available datasets
python download_datasets.py --list_only --repo_id DeepRetrieval/datasets

# Download all datasets
python download_datasets.py --repo_id DeepRetrieval/datasets --output_dir ./data

# Or download specific categories/datasets
python download_datasets.py --categories search_engine --datasets pubmed_32 --output_dir ./data
```
</details>


<details>
<summary style="font-weight: bold;">⛏️ Option 2: Process the data yourself</summary>

For example, for PubMed:
```bash
cd code
python download_datasets.py --categories raw_data --datasets pubmed --output_dir ./data
python data_preprocess/pubmed_32.py
```
(This will generate the required data structures in the appropriate format, but requires raw data access and more processing time.)
</details>


### **2. Get Your Search Engine API Key (required if use search engine)**

For example, for PubMed, you may get it following the instruction [here](https://support.nlm.nih.gov/kbArticle/?pn=KA-05317).

Then, put it in under `code/verl/utils/reward_score/apis/` as `pubmed_api.key`.


### **3. Reward function Related (optional)**

Reward Design (e.g., in `code/verl/utils/reward_score/pubmed.py`):


| Recall      | ≥ 0.7 | ≥ 0.5 | ≥ 0.4 | ≥ 0.3 | ≥ 0.1 | ≥ 0.05 | < 0.05 |
|-------------|-------|-------|-------|-------|-------|--------|--------|
| **Reward**  | +5.0  | +4.0  | +3.0  | +1.0  | +0.5  | +0.1   | -3.5   |



### **4. Customize Monitor Info (optional)**

modify `compute_reward_metrics()` in `code/verl/trainer/ppo/ray_trainer.py`


## 🏃 Run Training
```bash
conda activate zero
```

For example, for PubMed:
```bash
sh scripts/train/pubmed_32.sh 
```
(if you see Out-of-vram, try add `critic.model.enable_gradient_checkpointing=True` to the script)

### Think/Query Length During Training (PubMed)

![alt text](/images/length_study_horizontal.png "think length and query length during training")


## 🧐 Run Evaluation

```
sh scripts/eval/pubmed_32.sh
```
(You can run this script without training, as it will download our trained model from Huggingface by default)

**Results on Search Engines**

| Model | Method | Publication Recall | Clinical Trials Recall |
|-------|--------|-------------------|----------------------|
| Original Query | - | 10.36 | 18.01 |
| GPT-3.5 | - | 11.67 | 9.42 |
| | w/o reasoning | 18.68 | 13.94 |
| GPT-4o | - | 17.59 | 16.25 |
| | w/o reasoning | 19.72 | 14.26 |
| Claude-3-Haiku | - | 11.26 | 10.10 |
| | w/o reasoning | 20.92 | 24.68 |
| Claude-3.5-Sonnet | - | 20.94 | 18.33 |
| | w/o reasoning | 19.08 | 18.41 |
| Mistral-7B-Inst | - | 7.18 | 8.08 |
| LEADS-7B (SFT) | - | **24.68** | **32.11** |
| Qwen2.5-3B-Inst | - | 6.59 | 6.09 |
| | w/o reasoning | 9.46 | 7.97 |
| **DeepRetrieval-3B** | - | **✨ 65.07 ✨** | **✨ 63.18 ✨** |
| | w/o reasoning | 51.90 | 53.31 |

*Table: Comparison of different models and methods on publication search and clinical trials search tasks.

**Note:** LEADS-7B is a state-of-the-art literature mining LLM trained on 20K reviews and 400K publications [https://arxiv.org/pdf/2501.16255]

## 🤝 Acknowledgement

This implementation is mainly based on [verl](https://github.com/volcengine/verl) and [PySerini](https://github.com/castorini/pySerini). The base model during the experiment is [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B). We sincerely appreciate their contributions to the open-source community.

## 📚 Cite DeepRetrieval
Current version (will update the author list upon project completion):
```
@misc{jiang2025deepretrievalpowerfulquerygeneration,
      title={DeepRetrieval: Powerful Query Generation for Information Retrieval with Reinforcement Learning}, 
      author={Pengcheng Jiang},
      year={2025},
      eprint={2503.00223},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2503.00223}, 
}
```

```
@article{deepretrieval,
  title={DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models and Reinforcement Learning},
  author={Jiang, Pengcheng and Lin, Jiacheng and Cao, Lang and Tian, Runchu and Kang, SeongKu and Wang, Zifeng and Sun, Jimeng and Han, Jiawei},
  howpublished={\url{https://github.com/pat-jj/DeepRetrieval}},
  year={2025}
}
```

Thanks for your interests! 😊