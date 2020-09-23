# PyTorch book 

![pytorch](https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png)
handbook by



![ais](ais.png)
##  Introduction
This is an open source book by AI SUMMER, the goal is to help those who want and use PyTorch for deep learning development and research.

The technology of deep learning is developing rapidly, and PyTorch is constantly updated, and I will gradually improve the relevant content.

## Release Notes
As the PyTorch version changes, the tutorial version will be the same as the PyTorch version.

PyTorch has released the latest version 1.6.0.



# TODO
Organize chapters and notebooks

!? benchmarks on data loading, weight initialization, optimizers

Custom dataloaders

Templates for better coding in pytorch

Updates of tensorboard instead of tensorboardX

useful functions

Custom Losses and losses from numpy tensors (as in cython), networks types of fusion, 


## Table of Contents

### Chapter 1: Getting Started with PyTorch

1. [Introduction to PyTorch](1_getting_started/1.1-pytorch-introduction.md)
1. [PyTorch environment setup](1_getting_started/1.2-pytorch-installation.md)
1. [PyTorch Deep Learning: 60-minute quick start (official)](1_getting_started/1.3-deep-learning-with-pytorch-60-minute-blitz.md)  
    - [Tensor](1_getting_started/1.4._tensor_tutorial.ipynb)
    - [Autograd: Automatic Derivation](1_getting_started/1.6.autograd_tutorial.ipynb)
    - [Neural Network](3_CNN/3_neural_networks_tutorial.ipynb)
    - [Train a classifier](3_CNN/4_cifar10_tutorial.ipynb)
    - [Selected Reading: Data Parallel Processing (Multi-GPU)](2_Advanced_PyTorch/5_data_parallel_tutorial.ipynb)
1. [Related Resource Introduction](chapter1/1.4-pytorch-resource.md)

### Chapter 2 Basics
#### 2.1 PyTorch Basics
1. [Tensor](2_Advanced_PyTorch/2.1.1.advanced-tensor.ipynb)
1. [Automatic Derivation](2_Advanced_PyTorch/2.1.2-advnaced-autograd.ipynb)
1. [Neural Network Package nn and Optimizer optm](2_Advanced_PyTorch/2.1.3-advanced-build-neural-network.ipynb)
1. [Data loading and preprocessing](2_Advanced_PyTorch/2.1.4-pytorch-basics-data-loader.ipynb)
#### 2.2 Deep Learning Basics and Mathematical Principles

- [Deep learning basics and mathematical principles](2_Advanced_PyTorch/2.2-deep-learning-basic-mathematics.ipynb)

#### 2.3 Introduction to Neural Networks

- [Introduction to Neural Networks](2_Advanced_PyTorch/2.3-deep-learning-neural-network-introduction.ipynb) 
Note: This chapter will crash when opened locally with Microsoft Edge, please open Chrome Firefox to view

#### 2.4 Convolutional Neural Networks

- [Convolutional Neural Networks](3_CNN/2.4-cnn.ipynb)

#### 2.5 Recurrent Neural Networks

- [Recurrent Neural Networks](4_RNN/2.5-rnn.ipynb)

### Chapter 3 Practice
#### 3.1 Logistic regression binary classification

- [Logistic regression binary classification](1_getting_started/1.8.logistic-regression.ipynb)


#### 3.2 CNN: MNIST dataset handwritten digit recognition

- [CNN: Handwritten digit recognition of MNIST dataset](3_CNN/3.2-mnist.ipynb)

#### 3.3 RNN Examples: Predicting Cosine by Sin

- [RNN Example: Predicting Cos by Sin](4_RNN/3.3-rnn.ipynb)

### Chapter 4 Improvement
#### 4.1 Fine-tuning networks

- [Fine-tuning](5_Optimization/4.1-fine-tuning.ipynb)

- [Loss functions]()
- [Optimizers]()
#### 4.2 Visualization

- [visdom](5_Optimization/4.2.1-visdom.ipynb)

- [tensorboardx](5_Optimization/4.2.2-tensorboardx.ipynb)
(tensorboard support now)
- [Visual Understanding Convolutional Neural Network](5_Optimization/4.2.3-cnn-visualizing.ipynb)

#### 4.3 Fast.ai
- [Fast.ai](5_Optimization/4.3-fastai.ipynb)
#### 4.4 Training Skills

#### 4.5 Multi-GPU Parallel Training
- [Multi-GPU parallel computing](5_Optimization/4.5-multiply-gpu-parallel-training.ipynb)

### Chapter 5 Applications
#### 5.1 Introduction to Kaggle
[Introduction to Kaggle](chapter5/5.1-kaggle.md)
#### 5.2 Structured Data
[Pytorch processing structured data](chapter5/5.2-Structured-Data.ipynb)
#### 5.3 Computer Vision
[Fashion MNIST image classification](3_CNN/5.3-Fashion-MNIST.ipynb)
#### 5.4 Natural Language Processing
- [Transformers]()
- [RNN encoder-decoder]()
- [with attention]()
- [Language models]()

#### 5.5 GANs

#### 5.6 Action Recognition

#### 5.7 Segmentation 2D, 3D

#### 5.8 Medical Imaging, Health & AI

#### 5.5 Collaborative Filtering

### Chapter 6 Mobile , IoT
- [Raspberry Pi with Pytorch](pi/)
- [Export to mobile]()
- [ONNX]()

### Chapter 7 Appendix



Summary of common operations of transforms

pytorch's loss function summary

pytorch's optimizer summary

## License


![](https://i.creativecommons.org/l/by-nc-sa/3.0/88x31.png)

