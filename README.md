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

1. [Introduction to PyTorch](chapter1/1.1-pytorch-introduction.md)
1. [PyTorch environment setup](chapter1/1.2-pytorch-installation.md)
1. [PyTorch Deep Learning: 60-minute quick start (official)](chapter1/1.3-deep-learning-with-pytorch-60-minute-blitz.md)  
    - [Tensor](chapter1/1_tensor_tutorial.ipynb)
    - [Autograd: Automatic Derivation](chapter1/2_autograd_tutorial.ipynb)
    - [Neural Network](chapter1/3_neural_networks_tutorial.ipynb)
    - [Train a classifier](chapter1/4_cifar10_tutorial.ipynb)
    - [Selected Reading: Data Parallel Processing (Multi-GPU)](chapter1/5_data_parallel_tutorial.ipynb)
1. [Related Resource Introduction](chapter1/1.4-pytorch-resource.md)

### Chapter 2 Basics
#### 2.1 PyTorch Basics
1. [Tensor](chapter2/2.1.1.pytorch-basics-tensor.ipynb)
1. [Automatic Derivation](chapter2/2.1.2-pytorch-basics-autograd.ipynb)
1. [Neural Network Package nn and Optimizer optm](chapter2/2.1.3-pytorch-basics-nerual-network.ipynb)
1. [Data loading and preprocessing](chapter2/2.1.4-pytorch-basics-data-loader.ipynb)
#### 2.2 Deep Learning Basics and Mathematical Principles

- [Deep learning basics and mathematical principles](chapter2/2.2-deep-learning-basic-mathematics.ipynb)

#### 2.3 Introduction to Neural Networks

- [Introduction to Neural Networks](chapter2/2.3-deep-learning-neural-network-introduction.ipynb) 
Note: This chapter will crash when opened locally with Microsoft Edge, please open Chrome Firefox to view

#### 2.4 Convolutional Neural Networks

- [Convolutional Neural Networks](chapter2/2.4-cnn.ipynb)

#### 2.5 Recurrent Neural Networks

- [Recurrent Neural Networks](chapter2/2.5-rnn.ipynb)

### Chapter 3 Practice
#### 3.1 Logistic regression binary classification

- [Logistic regression binary classification](chapter3/3.1-logistic-regression.ipynb)


#### 3.2 CNN: MNIST dataset handwritten digit recognition

- [CNN: Handwritten digit recognition of MNIST dataset](chapter3/3.2-mnist.ipynb)

#### 3.3 RNN Examples: Predicting Cosine by Sin

- [RNN Example: Predicting Cos by Sin](chapter3/3.3-rnn.ipynb)

### Chapter 4 Improvement
#### 4.1 Fine-tuning networks

- [Fine-tuning](chapter4/4.1-fine-tuning.ipynb)

- [Loss functions]()
- [Optimizers]()
#### 4.2 Visualization

- [visdom](chapter4/4.2.1-visdom.ipynb)

- [tensorboardx](chapter4/4.2.2-tensorboardx.ipynb)
(tensorboard support now)
- [Visual Understanding Convolutional Neural Network](chapter4/4.2.3-cnn-visualizing.ipynb)

#### 4.3 Fast.ai
- [Fast.ai](chapter4/4.3-fastai.ipynb)
#### 4.4 Training Skills

#### 4.5 Multi-GPU Parallel Training
- [Multi-GPU parallel computing](chapter4/4.5-multiply-gpu-parallel-training.ipynb)

### Chapter 5 Applications
#### 5.1 Introduction to Kaggle
[Introduction to Kaggle](chapter5/5.1-kaggle.md)
#### 5.2 Structured Data
[Pytorch processing structured data](chapter5/5.2-Structured-Data.ipynb)
#### 5.3 Computer Vision
[Fashion MNIST image classification](chapter5/5.3-Fashion-MNIST.ipynb)
#### 5.4 Natural Language Processing
- Transformers
- RNN encoder-decoder
- with attention
- Language models

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

