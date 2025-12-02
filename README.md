# video-cliping-detection
视频恶意剪辑伪造分析

## 算法原理请参考

Gaussian-model-of-optical-flow

```
@inproceedings{wang2013identifying,
  title={Identifying video forgery process using optical flow},
  author={Wang, Wan and Jiang, Xinghao and Wang, Shilin and Wan, Meng and Sun, Tanfeng},
  booktitle={International Workshop on Digital Watermarking},
  pages={244--257},
  year={2013},
  organization={Springer}
}
```

Sum-of-X-and-Y-optical-flow

```
@inproceedings{chao2012novel,
  title={A novel video inter-frame forgery model detection scheme based on optical flow consistency},
  author={Chao, Juan and Jiang, Xinghao and Sun, Tanfeng},
  booktitle={International Workshop on Digital Watermarking},
  pages={267--281},
  year={2012},
  organization={Springer}
}
```

Hog-of-video-frames

```
@article{fadl2020exposing,
  title={Exposing video inter-frame forgery via histogram of oriented gradients and motion energy image},
  author={Fadl, Sondos and Han, Qi and Qiong, Li},
  journal={Multidimensional Systems and Signal Processing},
  volume={31},
  number={4},
  pages={1365--1384},
  year={2020},
  publisher={Springer}
}
```

## 环境

```
pip install opencv-python numpy matplotlib scipy scenedetect[opencv]
```

## 说明

首先使用scenedetect工具进行镜头切分，然后使用上述三个算法在切分是频段中进行分析。