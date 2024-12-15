# LiteSearch

This repository contains the code for the paper "LiteSearch: Efficacious Tree Search for LLM" (AAAI 2025).

## Getting Started

### Train a Value Network

1. **Obtain Llama-3-8B**: You can get the Llama-3-8B model from the [official website](https://llama.meta.com/) or [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B).

2. **Prepare Training Data**: Prepare your training data according to the guidelines provided in the paper. The output format should match the example shown in `train_demo.json`.

3. **Train the Model**:

```bash
Value/train.sh
```

### Run LiteSearch

1. **Setup Server**:

   - For the policy, use the latest version of VLLM.

   - For the value, run the provided script:

```bash
Value/run_hf.sh
```

2. **Run LiteSearch**:

```bash
LiteSearch/search_batch.py
```

Wait for the search to complete.

## Notes

All experiments in our paper are conducted on 8 V100 GPU (32G).
Ensure you have all necessary dependencies installed and configured before running the scripts. For more details on dependencies and setup, refer to the paper.

## Citation

```
@article{wang2024litesearch,
  title={Litesearch: Efficacious tree search for llm},
  author={Wang, Ante and Song, Linfeng and Tian, Ye and Peng, Baolin and Yu, Dian and Mi, Haitao and Su, Jinsong and Yu, Dong},
  journal={arXiv preprint arXiv:2407.00320},
  year={2024}
}
```
