# Restormer-Plus for Real World Image Deraining: the Runner-up Solution to the GT-RAIN Challenge (CVPR 2023 UG2+ Track 3)
This is the Python code used to implement the Restormer-Plus method as described in the technical report:

[**Restormer-Plus for Real World Image Deraining: One State-of-the-Art Solution to the GT-RAIN Challenge (CVPR 2023 UG2+ Track 3)**  
Chaochao Zheng, Luping Wang, Bin Liu](https://arxiv.org/abs/2305.05454)

[//]: # (## Technical Report Link)

[//]: # ([xx]&#40;xxx&#41;)

## Abstract
This technical report presents our Restormer-Plus approach, which was submitted to the GT-RAIN Challenge (CVPR 2023 UG$^2$+ Track 3). Details regarding the challenge are available at http://cvpr2023.ug2challenge.org/track3.html. Our Restormer-Plus outperformed all other submitted solutions in terms of peak signal-to-noise ratio (PSNR). It consists mainly of four modules: the single image de-raining module, the median filtering module, the weighted averaging module, and the post-processing module. We named the single-image de-raining module Restormer-X, which is built on Restormer and performed on each rainy image. The median filtering module is employed as a median operator for the 300 rainy images associated with each scene. The weighted averaging module combines the median filtering results with that of Restormer-X to alleviate overfitting if we only use Restormer-X. Finally, the post-processing module is used to improve the brightness restoration. Together, these modules render Restormer-Plus to be one state-of-the-art solution to the GT-RAIN Challenge. Our code is available at https://github.com/ZJLAB-AMMI/Restormer-Plus.

## Dataset
The dataset can be found [here](https://drive.google.com/drive/folders/1NSRl954QPcGIgoyJa_VjQwh_gEaHWPb8).

## Requirements

- einops==0.3.0
- natsort==8.3.1
- numpy==1.21.5
- opencv_contrib_python==4.2.0.32
- Pillow==9.2.0
- piq==0.7.0
- skimage==0.0
- tabulate==0.8.10
- torch==1.12.1
- torchvision==0.13.1

## Setup
Download the dataset from the link above and change the parameters in the ```train.py``` and ```test.py``` code to point to the appropriate directories (e.g., ```./gt-rain/```).

Download the pre-trained de-rain model from [link](https://drive.google.com/drive/folders/1ZEDDEVW0UgkpWi-N4Lj_JUoVChGXCu_u).

Install all the required packages.

## Running
**restormer-x:**

- training restormer baseline: set ```model_version=base``` and execute ```python /restormer_x/train.py```.

- training restormer+: set ```model_version=plus``` and execute ```python /restormer_x/train.py```.

- evaluate and/or test: execute ```python /restormer_x/test.py```.

**median:** execute ```python /median/median_derain.py```.

**ensemble:** execute ```python /ensemble/ensemble_derain.py```.

**post process:** execute ```python /post_process/post_process_derain.py```.

**submit result:** execute ```python repeat300.py```.

## Citation
If you find this code useful, please kindly cite  

@article{zheng2023RestormerPlus,

  title={Restormer-Plus for Real World Image Deraining: One State-of-the-Art Solution to the GT-RAIN Challenge (CVPR 2023 UG2+ Track 3)},
  
  author={Zheng, Chaochao, Wang, Luping and Liu, Bin},
  
  journal={arXiv preprint arXiv:2305.05454},
  
  year={2023}
  
}
## Disclaimer
Please only use the code and dataset for research purposes.

## Contact
Chaochao Zheng</br>
Zhejiang Lab, Research Center for Applied Mathematics and Machine Intelligence</br>
zhengcc@zhejianglab.com

Luping Wang</br>
Zhejiang Lab, Research Center for Applied Mathematics and Machine Intelligence</br>
wangluping@zhejianglab.com
