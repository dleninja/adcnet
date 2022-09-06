Automated Dispersion Compensation Network in Python using Keras
===============================================================

Overview
------------
Chromatic dispersion compensation is a common problem that degrades the resolution in optical coherence tomography (OCT).
In this project we present a deep learning approach for dispersion compensation using a fully convolutional network (FCN)
named automated dispersion compensation network (ADC-Net).

![The input in ADCNet is an array of size (X,Y,N), where N is the number of partially dipersion comepensated OCT B-scans, the output of ADCNet is a single fully dipersion compensated OCT B-scan.](https://github.com/dleninja/adcnet/blob/main/misc/pipeline.png?raw=true)

The ADC-Net is based on a redesigned UNet architecture which employs an encoder-decoder pipeline. The input section encompasses single or multichannel partially compensated OCT B-scans with individual retinal layers optimized. Corresponding output are fully compensated OCT B-scans with all retinal layers optimized. 

Dataset
------------
The dataset were collected by the Biomedical Optics and Ophthalmic Imaging Laboratory at the University of Illinois at Chicago. This study has been conducted in compliance with the ethical regulations reported in the Declaration of Helsinki and has been authorized by the institutional review board of the University of Illinois at Chicago.

![The Ground Truths were the fully dispersion compensated OCT B-scans and were prepared by stitching the partially dispersion compensated images together.](https://github.com/dleninja/adcnet/blob/main/misc/ground_truth.png?raw=true)

In this repository contains the training and testing dataset. A total of 9 OCT volumes were captured. Each OCT volume consists of ~1200 B-scans. Seven of these OCT volumes (8400 B-scans) were used for training the model, and another two (2400 B-scans) were used as testing set.

Model Architecture
------------
The ADC-Net is a FCN based on a modified UNet algorithm, which consists of an encoder-decoder architecture. The input to the ADC-Net can be of a single channel or a multichannel system. Each input is an OCT B-scan image which was compensated by different second-order dispersion compensation coefficients and hence the B-scans in each channel are optimally compensated at different layers or depths. The output is dispersion compensated OCT B-scans where all layers in different depths are compensated effectively. 

Dependencies
------------
- tensorflow >= 1.31.1
- keras >= 2.2.4
- python >= 3.7.1

Citations
------------
Ahmed, Shaiban, David Le, Taeyoon Son, Tobiloba Adejumo, Guangying Ma, and Xincheng Yao. "ADC-Net: An Open-Source Deep Learning Network for Automated Dispersion Compensation in Optical Coherence Tomography." Frontiers in Medicine 9 (2022).
[(article)](https://www.frontiersin.org/articles/10.3389/fmed.2022.864879/full)
