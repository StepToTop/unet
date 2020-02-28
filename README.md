# Implementation of deep learning framework -- Unet, using Keras (master version)

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

### Data

The original dataset is from [isbi challenge](http://brainiac2.mit.edu/isbi_challenge/), and I've downloaded it and done the pre-processing.

You can find it in folder data/membrane.

### Data augmentation

The data for training contains 30 512*512 images, which are far not enough to feed a deep learning neural network. I use a module called ImageDataGenerator in keras.preprocessing.image to do data augmentation.

See dataPrepare.ipynb and data.py for detail.


### Model

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 512*512 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

### Training

The model is trained for 5 epochs.

After 5 epochs, calculated accuracy is about 0.97.

Loss function for the training is basically just a binary crossentropy.


---
### Results

Use the trained model to do segmentation on test images, the result is statisfactory.

## Recurrent U-Net 的实现, using Keras

### 参考文献

[Wang W, Yu K, Hugonot J, et al. Recurrent U-Net for resource-constrained segmentation[C]//Proceedings of the IEEE International Conference on Computer Vision. 2019: 2142-2151.](https://arxiv.org/pdf/1906.04913.pdf)

### 使用训练平台

[谷歌colab](https://colab.research.google.com/)

### 优化网络结构