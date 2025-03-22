# VectorPainter: Advanced Stylized Vector Graphics Synthesis Using Stroke-Style Priors

[![ICME 2025](https://img.shields.io/badge/ICME%202025-Paper-6420AA?style=for-the-badge&logo=openreview&logoColor=white)](https://arxiv.org/abs/2405.02962)
[![ArXiv](https://img.shields.io/badge/arXiv-2405.02962-FF6347?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2405.02962)
[![Project Website](https://img.shields.io/badge/Website-Project%20Page-4682B4?style=for-the-badge&logo=github&logoColor=white)](https://hjc-owo.github.io/VectorPainterProject/)

This repository contains our official implementation of the ICME 2025 paper: "VectorPainter: Advanced Stylized Vector Graphics Synthesis Using Stroke-Style Priors." VectorPainter synthesizes text-guided vector graphics by imitating strokes.

![teaser](./image/teaser.png)

## 🔥 Latest Update

- _2025.03_: 🔥 We released the **code** for [VectorPainter](https://hjc-owo.github.io/VectorPainterProject/).
- _2025.03_: 🎉 VectorPainter accepted by ICME 2025. 🎉

## 📌 Installation Guide

### 🛠️ Step 1: Set Up the Environment

To quickly get started with **VectorPainter**, follow the steps below.  
These instructions will help you run **quick inference locally**.

Run the following command in the **top-level directory**:

```shell
chmod +x install.sh
bash install.sh
```

### 🛠️ Step 2: Download Pretrained Stable Diffusion Model

VectorPainter requires a pretrained Stable Diffusion (SD) model.

#### 🔄 Auto-Download (Recommended)

Set `model_download=True` in `/conf/config.yaml` before running VectorPainter.
Alternatively, append `model_download=True` to the execution script.

## 🎨 Quickstart

### Style: Starry Night by van Gogh

- sydney opera house
```shell
python vectorpainter.py x=stroke "prompt='A photo of Sydney opera house'" style="./assets/starry.jpg" canvas_w=1024 canvas_h=1024 result_path='./workspace/starry/sydney_opera_house'
```

- mountain and cloud
```shell
python vectorpainter.py x=stroke "prompt='A mountain, with clouds in the sky'" style="./assets/starry.jpg" canvas_w=1024 canvas_h=1024 result_path='./workspace/starry/mountain'
```

### Style: Pink Cloud (Le nuage rose by Paul Signac)

- Sakura tree
```shell
python vectorpainter.py x=stroke "prompt='sakura tree.'" style="./assets/starry.jpg" x.num_paths=20000 canvas_w=1024 canvas_h=1024 result_path='./workspace/pink_cloud/sakura'
```
## 📚 Acknowledgement

The project is built based on the following repository:

- [BachiLi/diffvg](https://github.com/BachiLi/diffvg)
- [huggingface/diffusers](https://github.com/huggingface/diffusers)
- [ximinng/DiffSketcher](https://github.com/ximinng/DiffSketcher)
- [ximinng/PyTorch-SVGRender](https://github.com/ximinng/PyTorch-SVGRender)

We gratefully thank the authors for their wonderful works.

## 📎 Citation

If you use this code for your research, please cite the following work:

```
@article{hu2024vectorpainter,
  title={VectorPainter: Advanced Stylized Vector Graphics Synthesis Using Stroke-Style Priors},
  author={Hu, Juncheng and Xing, Ximing and Zhang, Jing and Yu, Qian},
  journal={arXiv preprint arXiv:2405.02962},
  year={2024}
}
```

## ©️ Licence

This work is licensed under a MIT License.
