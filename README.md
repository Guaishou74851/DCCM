# Compressive Confocal Microscopy Imaging at the Single-Photon Level with Ultra-Low Sampling Ratios (Communications Engineering 2024) [PyTorch]

[![icon](https://img.shields.io/badge/Nature-Paper-<COLOR>.svg)](https://www.nature.com/articles/s44172-024-00236-x) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=Guaishou74851.DCCM)

Shuai Liu\*, [Bin Chen](https://scholar.google.com/citations?hl=en&user=aZDNm98AAAAJ)\*, Wenzhen Zou, [Hao Sha](https://scholar.google.com/citations?user=-mqUZ8oAAAAJ), Xiaochen Feng, [Sanyang Han](https://www.sigs.tsinghua.edu.cn/hsy/main.htm), [Xiu Li](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ), Xuri Yao, [Jian Zhang](https://jianzhang.tech/)†, and [Yongbing Zhang](https://scholar.google.com/citations?user=0KlvTEYAAAAJ)†

*Tsinghua Shenzhen International Graduate School, Tsinghua University, Shenzhen, China.*

*School of Electronic and Computer Engineering, Peking University, Shenzhen, China.*

*School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen), Shenzhen, China.*

*Center for Quantum Technology Research, School of Physics, Beijing Institute of Technology, Beijing, China.*

Accepted for publication in [Communications Engineering](https://www.nature.com/commseng/) (Nature Communications) 2024.

\* Equal contribution   † Corresponding author

## Abstract

Laser-scanning confocal microscopy serves as a critical instrument for microscopic research in biology. However, it suffers from low imaging speed and high phototoxicity. Here we build a novel deep compressive confocal microscope, which employs a digital micromirror device as a coding mask for single-pixel imaging and a pinhole for confocal microscopic imaging respectively. Combined with a deep learning reconstruction algorithm, our system is able to achieve high-quality confocal microscopic imaging with low phototoxicity. Our imaging experiments with fluorescent microspheres demonstrate its capability of achieving single-pixel confocal imaging with a sampling ratio of only approximately 0.03% in specific sparse scenarios. Moreover, the deep compressive confocal microscope allows single-pixel imaging at the single-photon level, thus reducing the excitation light power requirement for confocal imaging and suppressing the phototoxicity. We believe that our system has great potential for long-duration and high-speed microscopic imaging of living cells.

## Overview

![overview](figs/overview.png)

## Environment

```shell
torch.__version__ == '2.2.1+cu121'
numpy.__version__ == '1.24.4'
skimage.__version__ == '0.21.0'
```

## Data and Pretrained Model Weights

Download the [data](https://drive.google.com/file/d/1FCVwqjb8_J-yTc47t1E0mF8TlM-NdMp5/view) and [pretrained model weights](https://drive.google.com/file/d/1tHohEMx35Dg5qh8X-15CQesx6Q0mDTpv/view). Unzip the files into `./data` and `./code/weight` directories, respectively.

The paths of all files should be:

```
.
├── README.md
├── code
│   ├── model.py
│   ├── test.py
│   ├── test.sh
│   ├── train.py
│   ├── train.sh
│   ├── utils.py
│   └── weight
│       ├── f-actin
│       │   └── layer_9_f_128
│       │       └── net_params_30000.pkl
│       ├── flureoscent_microsphere
│       │   └── layer_9_f_128
│       │       └── net_params_30000.pkl
│       ├── nucleus
│       │   └── layer_9_f_128
│       │       └── net_params_30000.pkl
│       └── potato_tuber
│           └── layer_9_f_128
│               └── net_params_30000.pkl
├── data
│   ├── A_128.npy
│   ├── A_32.npy
│   ├── f-actin
│   │   ├── test_X.npy
│   │   ├── test_X_WF.npy
│   │   ├── test_Y128.npy
│   │   ├── test_Y32.npy
│   │   ├── train_X.npy
│   │   ├── train_X_WF.npy
│   │   ├── train_Y128.npy
│   │   └── train_Y32.npy
│   ├── flureoscent_microsphere
│   │   ├── test_X.npy
│   │   ├── test_X_WF.npy
│   │   ├── test_Y128.npy
│   │   ├── test_Y32.npy
│   │   ├── train_X.npy
│   │   ├── train_X_WF.npy
│   │   ├── train_Y128.npy
│   │   └── train_Y32.npy
│   ├── nucleus
│   │   ├── test_X.npy
│   │   ├── test_X_WF.npy
│   │   ├── test_Y128.npy
│   │   ├── test_Y32.npy
│   │   ├── train_X.npy
│   │   ├── train_X_WF.npy
│   │   ├── train_Y128.npy
│   │   └── train_Y32.npy
│   └── potato_tuber
│       ├── test_X.npy
│       ├── test_X_WF.npy
│       ├── test_Y128.npy
│       ├── test_Y32.npy
│       ├── train_X.npy
│       ├── train_X_WF.npy
│       ├── train_Y128.npy
│       └── train_Y32.npy
└── figs
    └── overview.png
```

## Test

```shell
cd code
python test.py --data_type=nucleus
python test.py --data_type=flureoscent_microsphere
python test.py --data_type=f-actin
python test.py --data_type=potato_tuber
```

The reconstructed images will be in `./code/result`.

## Train

```shell
cd code
python train.py --data_type=nucleus
python train.py --data_type=flureoscent_microsphere
python train.py --data_type=f-actin
python train.py --data_type=potato_tuber
```

The log and model files will be in `./code/log` and `./code/weight`, respectively.

## Citation

If you find the code helpful in your research or work, please cite the following paper:

```latex
@article{liu2024compressive,
  title={Compressive confocal microscopy imaging at the single-photon level with ultra-low sampling ratios},
  author={Liu, Shuai and Chen, Bin and Zou, Wenzhen and Sha, Hao and Feng, Xiaochen and Han, Sanyang and Li, Xiu and Yao, Xuri and Zhang, Jian and Zhang, Yongbing},
  journal={Communications Engineering},
  volume={3},
  number={1},
  pages={88},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
