# light-weight-depth-estimation
<br>

This repo provides trained models and evaluation code for the light weight model for depth estimation.

Junjie Hu, Chenyou Fan, Hualie Jiang, Xiyue Guo, Yuan Gao, Xiangyong Lu, Tin Lun Lam https://arxiv.org/abs/2105.06143



Results
-
![](https://github.com/junjH/light-weight-depth-estimation/raw/master/figs/results.png)
![](https://github.com/junjH/light-weight-depth-estimation/raw/master/figs/visualization.png)


Dependencies
-
+ python 3.6<br>
+ Pytorch 1.7.1<br>


Running
-
Download the NYU-v2 dataset: [NYU-v2 dataset](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view?usp=sharing) <br>

Download the trained networks for depth estimation :[Depth estimation networks](https://drive.google.com/file/d/1yr5S5FIheL1mUfBzVJ8KqcIq9JP-jd4z/view?usp=sharing) <br>

+ ### Test<br>
  python test.py<br>

Citation
-
  @article{hu2021boosting,
    title={Boosting Light-Weight Depth Estimation Via Knowledge Distillation},
    author={Hu, Junjie and Fan, Chenyou and Jiang, Hualie and Guo, Xiyue and Gao, Yuan and Lu, Xiangyong and Lam, Tin Lun},
    journal={arXiv preprint arXiv:2105.06143},
    year={2021}
  }
