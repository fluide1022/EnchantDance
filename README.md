# EnchantDance: Unveiling the Potential of Music-Driven Dance Movement

<p align="center">
  <a href='https://arxiv.org/abs/2312.15946'>
    <img src='https://img.shields.io/badge/Arxiv-2312.06553-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://fluide1022.github.io/EnchantDance/'>
    <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20Chrome&logoColor=white'>
  </a>
</p>


<p align="center">
<!-- <h1 align="center">InterDiff: Generating 3D Human-Object Interactions with Physics-Informed Diffusion</h1> -->
<strong>EnchantDance: Unveiling the Potential of Music-Driven Dance Movement</strong></h1>
   <p align="center">
    <a href='https://scholar.google.com/citations?user=5XsDL6kAAAAJ&hl=zh-CN' target='_blank'>Bo Han</a>&emsp;
    <a href='' target='_blank'>Teng Zhang</a>&emsp;
    <a href='' target='_blank'>Zeyu Ling</a>&emsp;
    <a href='https://rayeren.github.io/' target='_blank'>Yi Ren</a>&emsp;
    <a href='https://scholar.google.com/citations?user=e6_J-lEAAAAJ&hl=en' target='_blank'>Xiang Yin</a>&emsp;
    <a href='https://feilinh.cn/' target='_blank'>Feilin Han</a>&emsp;
    <br>
    Zhejiang University;    Bytedance AI Lab;
    Beijing Film Academy
    <br>
  </p>
</p>

## 🛠️ Environment

| Package | Version |
|---------|---------|
| Python  | 3.9.16  |
| PyTorch | 1.12.1+cu113 |
| NumPy   | 1.24.2  |

# Train
## 🏃 Training

### Train VAE
python -m train --cfg configs/config_vae.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug

### Train Diffusion

python -m train --cfg configs/config_diffusion.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug

# 🎯 Inference
## 🎵 Music to Dance Generation
```bash
python demo.py --task=music_dance
```

## 🎨 Visualization Examples
- 💃 Motion sequences will be saved as `.npz` files
- 🎦 Rendered videos to `.mp4` files with Blender

## 🤝 Citation

If you find this repository useful for your work, please consider citing it as follows:

```bibtex
@article{han2023enchantdance,
  title={EnchantDance: Unveiling the Potential of Music-Driven Dance Movement},
  author={Han, Bo and Zhang, Teng and Ling, Zeyu and Ren, Yi and Yin, Xiang and Han, Feilin},
  journal={arXiv preprint arXiv:2312.15946},
  year={2023}
}
```