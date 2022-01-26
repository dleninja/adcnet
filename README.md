Automated Dispersion Compensation Network in Python using Keras
===============================================================

Overview
------------
Chromatic dispersion compensation is a common problem that degrades the resolution in optical coherence tomography (OCT).
In this project we present a deep learning approach for dispersion compensation using a fully convolutional network (FCN)
named automated dispersion compensation network (ADC-Net).

![The input in ADCNet is an array of size (X,Y,N), where N is the number of partially dipersion comepensated OCT B-scans, the output of ADCNet is a single fully dipersion compensated OCT B-scan.](https://github.com/dleninja/adcnet/blob/main/misc/pipeline.png?raw=true)

Dataset
------------
The dataset were collected by the Biomedical Optics and Ophthalmic Imaging Laboratory at the University of Illinois at Chicago. This study has been conducted in compliance with the ethical regulations reported in the Declaration of Helsinki and has been authorized by the institutional review board of the University of Illinois at Chicago.

![The Ground Truths were the fully dispersion compensated OCT B-scans and were prepared by stitching the partially dispersion compensated images together.](https://github.com/dleninja/adcnet/blob/main/misc/ground_truth_preparation.png?raw=true)

In this repository contains the training and testing dataset.

Dependencies
------------
- tensorflow >= 2.3.0
- keras >= 2.4.0
- python >= 3.8

Citations
-----------
Shaiban Ahmed, David Le, Taeyoon Son, Tobiloba Adejumo, and Xincheng Yao.
"ADC-Net: An Open-Source Deep Learning Network for Automated Dispersion Compensation in Optical Coherence Tomography", Frontiers in Medicine (2022).
