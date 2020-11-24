# Isleyen-PhD-Research
This repository contains the scripts used in the thesis "Development of Artificial-Intelligence Based Autonomous Roof Fall Hazard Detection System" by Ergin Isleyen.
Mining Engineering Department, Colorado School of Mines.

***CNN_TransferLearning.py***
Training and validation of a ResNet-152 Convolutional Neural Network pre-trained on ImageNet dataset.
Language: Python
Library: pytorch, torchvision, numpy, matplotlib, sklearn
Input: input image directory
  structure of the input image directory
  -input
  --train
  ---hazardous
  ---nonhazardous
  --val
  ---hazardous
  ---nonhazardous
  --test
  ---hazardous
  ---nonhazardous

***DL_interpretation.py***
Deep learning interpretation using integrated gradients technique and model testing.
Language: Python
Library: pytorch, captum, numpy, matplotlib
Input: saved network file, directory of test images
  structure of the test image directory
  -test
  --hazardous
  --nonhazardous

***GaborFilter.m***
Runs a gabor filter and calculates the variance of the filtered image.
Language: Matlab
Library: Image Processing Toolbox, Statistics and Machine Learning Toolbox
Input: Directory of tehe image files
