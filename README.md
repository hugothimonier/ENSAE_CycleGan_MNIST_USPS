# ENSAE CycleGan MNIST USPS

This repo contains the work of <b> Hugo Thimonier </b> (ENSAE, ENS Paris Saclay) and <b> Gabriel Kasmi </b> (ENSAE, ENS Paris Saclay) on applying the Cycle GAN using PyTorch model to estimate the mapping from the MNIST digit distribution to the USPS digit distribution.

### Link to google colab : https://colab.research.google.com/drive/1cYCBRVjDbN2ffqbYnG4grgKTP0bB2n6i#scrollTo=Zinj7hAGpCai

## Usage

#### Clone the repository

```bash
$ git clone https://github.com/hugothimonier/ENSAE_CycleGan_MNIST_USPS.git
$ cd ENSAE_CycleGan_MNIST_USPS/
```

#### Train Model

```bash
python main.py
```


## Model

The generator considered is the following 

<p align="center">
  <img src="https://github.com/hugothimonier/ENSAE_CycleGan_MNIST_USPS/blob/master/img/Generator.png">
</p>


The Discriminator considered is the following

<p align="center">
  <img src="https://github.com/hugothimonier/ENSAE_CycleGan_MNIST_USPS/blob/master/img/Discriminator.png">
</p>

The overall model is as follows
<p align="center">
  <img src="https://github.com/hugothimonier/ENSAE_CycleGan_MNIST_USPS/blob/master/img/cyclegan.png" height = '60%' width ='60%'>
</p>

## Sample and Results

The MINST dataset and USPS dataset are as follows

<p align="center">
  <img src="https://github.com/hugothimonier/ENSAE_CycleGan_MNIST_USPS/blob/master/img/mnist_sample.png" alt="MNIST" height = '30%' width ='30%' />
  <img src="https://github.com/hugothimonier/ENSAE_CycleGan_MNIST_USPS/blob/master/img/usps_sample.png" alt="USPS" height = '30%' width ='30%' /> 
</p>

The objective is to translate MNIST images into USPS images. The two main differences between the images is the cropping and the blurriness. Our model produced the following results after 1000 iterations.

<p align="center">
  <img src="https://github.com/hugothimonier/ENSAE_CycleGan_MNIST_USPS/blob/master/results/sample-001000-X-Y.png" alt="MNIST to USPS" height = '30%' width ='30%' />
  <img src="https://github.com/hugothimonier/ENSAE_CycleGan_MNIST_USPS/blob/master/results/sample-001000-Y-X.png" alt="USPS to MNIST" height = '30%' width ='30%' /> 
</p>


## Repo description

### .py files

### img
Images used for the ReadMe.md

### results
Images of generated pictures
