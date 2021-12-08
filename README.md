# LISTA-CE
#### Adaptive Channel Estimation Based on Model-Driven Deep Learning for Wideband mmWave Systems, GLOBECOM 2021 (Tensorflow Code)

This repository is for LISTA-CE introduced in the following paper, which is accepted by GLOBECOM 2021:

W. Jin, H. He, C.-K. Wen, S. Jin, and G. Y. Li, “Adaptive Channel Estimation Based on Model-Driven Deep Learning for Wideband mmWave Systems” [[pdf]](https://arxiv.org/pdf/2104.13656.pdf).

Training data can be downloaded at [GoogleDrive](https://drive.google.com/drive/folders/1OeRStZSpSX7V3PTSwgqUjdhIQ6qlW3dG?usp=sharing). 

The code is tested on Linux environments (Tensorflow: 1.14.0, CUDA10.1).

#### Files Description:
###### LISTA_CE_SingleNet.py: Used for training and testing LISTA-CE in the paper.
###### LISTA_CE_Hyper_averModel.py: Used for training and testing LISTA-CEAver in the paper.
###### LISTA_CE_Hyper_TrainJointly.py: Used for training and testing LISTA-CEHyper in the paper.
###### 'LISTA_CE_Hyper_averModel.py' should be run before 'LISTA_CE_Hyper_TrainJointly.py' to get the LISTA-CEAver network, as described in the first paragraph of page 5 in the paper.
###### For the convenience of code writing, one layer of network in the paper is represented as two layers in the code. The two are equivalent.

## Introduction
Channel estimation in wideband millimeter-wave (mmWave) systems is very challenging due to the beam squint effect. To solve the problem, we propose a learnable iterative shrinkage thresholding algorithm-based channel estimator (LISTA-CE) based on deep learning. The proposed channel estimator can learn to transform the beam-frequency mmWave channel into the domain with sparse features through training data. The transform domain enables us to adopt a simple denoiser with few trainable parameters. We further enhance the adaptivity of the estimator by introducing hypernetwork to automatically generate learnable parameters for LISTA-CE online. Simulation results show that the proposed approach can significantly outperform the state-of-the-art deep learning-based algorithms with lower complexity and fewer parameters and adapt to new scenarios rapidly.

![LISTA-CE](/Figs/LISTA_CE.png)
Figure 1. Illustration of our proposed LISTA-CE framework.

![LISTA-CE](/Figs/LISTA_CEHyper.png)
Figure 2. Illustration of the *t*-th layer of our proposed LISTA-CEHyper framework.

## Citation
If you find our code helpful in your resarch or work, please cite our paper.
```
@misc{jin2021adaptive,
      title={Adaptive Channel Estimation Based on Model-Driven Deep Learning for Wideband mmWave Systems}, 
      author={Weijie Jin and Hengtao He and Chao-Kai Wen and Shi Jin and Geoffrey Ye Li},
      year={2021},
      eprint={2104.13656},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}
