# Generative Recommendation with Semantic IDs (GRID)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/config-hydra-89b8cd)](https://hydra.cc/)
[![Lightning](https://img.shields.io/badge/pytorch-lightning-792ee5)](https://lightning.ai/)
[![arXiv](https://img.shields.io/badge/arXiv-2507.22224-b31b1b.svg)](https://arxiv.org/abs/2507.22224)


**GRID** (Generative Recommendation with Semantic IDs) is a state-of-the-art framework for generative recommendation systems using semantic IDs, developed by a group of scientists and engineers from [Snap Research](https://research.snap.com/team/user-modeling-and-personalization.html). This project implements novel approaches for learning semantic IDs from text embedding and generating recommendations through transformer-based generative models.

## 🚀 Overview

GRID facilitates generative recommendation three overarching steps:

- **Embedding Generation with LLMs**: Converting item text into embeddings using any LLMs available on Huggingface. 
- **Semantic ID Learning**: Converting item embedding into hierarchical semantic IDs using Residual Quantization techniques such as RQ-KMeans, RQ-VAE, RVQ. 
- **Generative Recommendations**: Using transformer architectures to generate recommendation sequences as semantic ID tokens. 


## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/snap-research/GRID.git
cd GRID

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Data Preparation

Prepare your dataset in the expected format:
```
data/
├── train/       # training sequence of user history 
├── validation/  # validation sequence of user history 
├── test/        # testing sequence of user history 
└── items/       # text of all items in the dataset
```

We provide pre-processed Amazon data explored in the [P5 paper](https://arxiv.org/abs/2203.13366) [4]. The data can be downloaded from this [google drive link](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view?usp=sharing).

### 2. Embedding Generation with LLMs

Generate embeddings from LLMs, which later will be transformed into semantic IDs. 

```bash
python -m src.inference experiment=sem_embeds_inference_flat data_dir=data/amazon_data/beauty # avaiable data includes 'beauty', 'sports', and 'toys'
```

### 3. Train and Generate Semantic IDs

Learn semantic ID centroids for embeddings generated in step 2:

```bash
python -m src.train experiment=rkmeans_train_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<output_path_from_step_2>/merged_predictions_tensor.pt \ # this can be found in the log dirs in step2
    embedding_dim=2048 \ # the model dimension of the LLMs you use in step 2. 2048 for flan-t5-xl as used in this example.
    num_hierarchies=3 \  # we train 3 codebooks
    codebook_width=256 \ # each codebook has 256 rows of centroids  
```

Generate SIDs:

```bash
python -m src.inference experiment=rkmeans_inference_flat \
    data_dir=data/amazon_data/beauty \
    embedding_path=<output_path_from_step_2>/merged_predictions_tensor.pt \ 
    embedding_dim=2048 \ 
    num_hierarchies=3 \  
    codebook_width=256 \ 
    ckpt_path=<the_checkpoint_you_just_get_above> # this can be found in the log dir for training SIDs
```


### 4. Train Generative Recommendation Model with Semantic IDs

Train the recommendation model using the learned semantic IDs:

```bash
python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/beauty \ 
    semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4 # Please note that we add 1 for num_hierarchies because in the previous step we appended one additional digit to de-duplicate the semantic IDs we generate.
```

### 4. Generate Recommendations

Run inference to generate recommendations:

```bash
python -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/beauty \ 
    semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
    ckpt_path=<the_checkpoint_you_just_get_above> \ # this can be found in the log dir for training GR models
    num_hierarchies=4 \ # Please note that we add 1 for num_hierarchies because in the previous step we appended one additional digit to de-duplicate the semantic IDs we generate.
```

### Decoder-Only Model (Extension)

In addition to the default encoder-decoder (T5-style) recommender, this repo
provides a **decoder-only (GPT-style)** generative recommender,
`SemanticIDDecoderOnly`. It is enabled entirely through the config (no code
changes are needed) by selecting the decoder-only experiment instead of the
default `tiger_*_flat` one:

```bash
# Train the decoder-only model
python -m src.train experiment=tiger_decoder_only_train_flat \
    data_dir=data/amazon_data/beauty \
    semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
    num_hierarchies=4 \
    model.loss_on_all_positions=false   # see loss toggle below
```

```bash
# Generate recommendations with the decoder-only model
python -m src.inference experiment=tiger_decoder_only_inference_flat \
    data_dir=data/amazon_data/beauty \
    semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
    ckpt_path=<the_checkpoint_you_just_get_above> \
    num_hierarchies=4
```

**Loss objective toggle.** The decoder-only model exposes a next-token
prediction at every position, so it supports two training objectives via the
`model.loss_on_all_positions` flag (default `true` in the experiment config):

- `model.loss_on_all_positions=false`: apply the loss only on the final target
  item (**loss-last**), matching the encoder-decoder TIGER objective.
- `model.loss_on_all_positions=true`: apply the full causal next-token loss on
  **all positions** (**loss-all**).

All other knobs (Semantic IDs, model dimensions, optimizer, beam search) are
shared with the encoder-decoder baseline, so switching the `experiment` value is
sufficient to compare architectures under identical settings.

## Supported Models:

### Semantic ID:

1. Residual K-means proposed in One-Rec [2]
2. Residual Vector Quantization
3. Residual Quantization with Variational Autoencoder [3]

### Generative Recommendation:

1. TIGER [1] (encoder-decoder)
2. Decoder-only (GPT-style) variant with selectable loss-last / loss-all objective

## 📚 Citation

If you use GRID in your research, please cite:

```bibtex
@inproceedings{grid,
  title     = {Generative Recommendation with Semantic IDs: A Practitioner's Handbook},
  author    = {Ju, Clark Mingxuan and Collins, Liam and Neves, Leonardo and Kumar, Bhuvesh and Wang, Louis Yufeng and Zhao, Tong and Shah, Neil},
  booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2025}
}
```

## 🤝 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- Configuration management by [Hydra](https://hydra.cc/)
- Inspired by recent advances in generative AI and recommendation systems
- Part of this repo is built on top of https://github.com/ashleve/lightning-hydra-template

## 📞 Contact

For questions and support:
- Create an issue on GitHub
- Contact the development team: Clark Mingxuan Ju (mju@snap.com), Liam Collins (lcollins2@snap.com), Bhuvesh Kumar (bhuvesh@snap.com) and Leonardo Neves (lneves@snap.com).

## Bibliography 

[1] Rajput, Shashank, et al. "Recommender systems with generative retrieval." Advances in Neural Information Processing Systems 36 (2023): 10299-10315.

[2] Deng, Jiaxin, et al. "Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment." arXiv preprint arXiv:2502.18965 (2025).

[3] Lee, Doyup, et al. "Autoregressive image generation using residual quantization." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

[4] Geng, Shijie, et al. "Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5)." Proceedings of the 16th ACM conference on recommender systems. 2022.
