<!DOCTYPE html>
<html>
<head>
<style>
* {
  box-sizing: border-box;
}

.column {
  float: left;
  width: 33.33%;
  padding: 5px;
}

/* Clearfix (clear floats) */
.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>
</head>
<body>

# ENSAE CycleGan MNIST USPS

This repo contains the work of <b> Hugo Thimonier </b> (ENSAE, ENS Paris Saclay) and <b> Gabriel Kasmi </b> (ENSAE, ENS Paris Saclay) on applying the Cycle GAN model to estimate the mapping from the MNIST digit distribution to the USPS digit distribution.

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
  <img src="https://github.com/hugothimonier/ENSAE_CycleGan_MNIST_USPS/blob/master/img/cyclegan.png">
</p>

## Sample

The MINST dataset and USPS dataset are as follows

<div class="row">
  <div class="column">
    <img src="https://github.com/hugothimonier/ENSAE_CycleGan_MNIST_USPS/blob/master/img/mnist_sample.png" alt="MNIST" height = '30%' width ='30%' >
  </div>
  <div class="column">
    <img src="https://github.com/hugothimonier/ENSAE_CycleGan_MNIST_USPS/blob/master/img/usps_sample.png" alt="USPS" height = '30%' width ='30%' >
  </div>
</div>

## Repo description
