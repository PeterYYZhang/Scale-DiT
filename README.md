# Scale-DiT: Ultra-High-Resolution Image Generation with Hierarchical Local Attention

[![arXiv](https://img.shields.io/badge/arXiv-2510.16325-b31b1b.svg)](https://arxiv.org/abs/2510.16325)

Official code for **Scale-DiT** ([arXiv:2510.16325](https://arxiv.org/abs/2510.16325)):
*Yuyao Zhang, Yu-Wing Tai*.


![Scale-DiT teaser](images/Teasor-2_page-001.png)


## Abstract
Ultra-high-resolution text-to-image generation demands both fine-grained texture synthesis and globally coherent structure, yet current diffusion models remain constrained to sub-1K×1K resolutions due to the prohibitive quadratic complexity of attention and the scarcity of native 4K training data. We present **Scale-DiT**, a new diffusion framework that introduces hierarchical local attention with low-resolution global guidance, enabling efficient, scalable, and semantically coherent image synthesis at ultra-high resolutions. Specifically, high-resolution latents are divided into fixed-size local windows to reduce attention complexity from quadratic to near-linear, while a low-resolution latent equipped with scaled positional anchors injects global semantics. A lightweight LoRA adaptation bridges global and local pathways during denoising, ensuring consistency across structure and detail. To maximize inference efficiency, we repermute token sequence in Hilbert curve order and implement a fused-kernel for skipping masked operations, resulting in a GPU-friendly design. Extensive experiments demonstrate that Scale-DiT achieves more than 2× faster inference and lower memory usage compared to dense attention baselines, while reliably scaling to 4K×4K resolution without requiring additional high-resolution training data. On both quantitative benchmarks (FID, IS, CLIP Score) and qualitative comparisons, Scale-DiT delivers superior global coherence and sharper local detail, matching or outperforming state-of-the-art methods that rely on native 4K training. Taken together, these results highlight hierarchical local attention with guided low-resolution anchors as a promising and effective approach for advancing ultra-high-resolution image generation.

## Pipeline

![Scale-DiT pipeline](images/Pipeline_new_page-001.png)

## 🔥 Highlights

- **Ultra-high-resolution** T2I generation up to **4K × 4K** without native 4K training data.
- **Hierarchical local attention** (near-linear scaling) + **low-res global guidance** via scaled positional anchors.
- **LoRA bridge** between global/local pathways during denoising.
- **Very sparse attention** to speed up inference.

## 🎬 Image demos 

Below are example generations with the comparison of high-res output and low-res output after bicubic interpolation.

| Prompt | High-res output | Low-res (after) |
|---|---|---|
| Realistic Persian cat relaxing on a velvet sofa, soft indoor lighting | ![](<images/lora_0_Realistic Persian cat relaxing on a velvet sofa, s_15.jpg>) | ![](<images/lora_0_Realistic Persian cat relaxing on a velvet sofa, s_15_low_res_after.jpg>) |
| serene mountain village nestled in the Swiss Alps, traditional wooden chalets with flower boxes, cobblestone paths winding between houses, snow-capped peaks in the background, golden hour lighting, smoke rising from chimneys, villagers in traditional clothing walking the streets, cozy warm atmosphere, detailed architecture, rustic charm. | ![](<images/lora_0_serene mountain village nestled in the Swiss Alps,_9.jpg>) | ![](<images/lora_0_serene mountain village nestled in the Swiss Alps,_9_low_res_after.jpg>) |
| Ultra-realistic portrait of an elderly man with deep wrinkles, soft window light, 85mm lens, shallow depth of field | ![](<images/lora_0_Ultra-realistic portrait of an elderly man with de_10.jpg>) | ![](<images/lora_0_Ultra-realistic portrait of an elderly man with de_10_low_res_after.jpg>) |

## Benchmark

> **Placeholder**: add quantitative tables (FID / IS / CLIPScore / speed / memory) here.

![Benchmark placeholder](https://dummyimage.com/1600x900/eeeeee/555555.png&text=Benchmark+%2F+Table+Placeholder)

## Visualization

Below are qualitative comparisons against other methods:

![4K comparison 1](images/Appendix-4k_page-001.png)

![4K comparison 2](images/Appendix-4k-2_page-001.png)

## 📑 Open-source plan

- Training code ✅
- Inference code ✅
- Model checkpoints (LoRA) ✅ (see `checkpoints/`)
- More evaluation scripts / benchmark tables (TODO)

## Installation

This project uses **Accelerate** + **Lightning** + **Diffusers**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch matching your CUDA (example only):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install accelerate lightning diffusers transformers peft safetensors datasets pyyaml pillow opencv-python psutil prodigyopt wandb
```

## Training scripts

Entrypoint: `src/train/train.py`

This repo is currently structured to generate sample images during a Lightning run (via `src/train/callbacks.py`).

## Inference (quickstart)

Use the provided config:

- `train/config/inference.yaml`

Run:

```bash
export XFL_CONFIG="./train/config/inference.yaml"
accelerate launch -m src.train.train --disable_wandb
```

Outputs:

```text
runs/<run_name><timestamp>/
  config.yaml
  output/   (images)
  ckpt/     (saved LoRA weights)
```

## Repo layout

- `src/flux/`: pipeline + transformer + generation
- `src/train/`: Lightning module, dataset wrapper, callbacks
- `train/config/`: example configs/prompts
- `train/script/`: convenience launch scripts

## Security / secrets

- Don’t commit tokens (e.g. `WANDB_API_KEY`). Use environment variables instead.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{zhang2025scaledit,
  title        = {Scale-DiT: Ultra-High-Resolution Image Generation with Hierarchical Local Attention},
  author       = {Yuyao Zhang and Yu-Wing Tai},
  year         = {2025},
  eprint       = {2510.16325},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  doi          = {10.48550/arXiv.2510.16325},
  url          = {https://arxiv.org/abs/2510.16325}
}
```

## License

Add your license here (e.g. MIT/Apache-2.0) and note any upstream model license constraints.


