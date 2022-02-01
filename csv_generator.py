"""
Automated Dispersion Compensation Network (ADC-Net).

This file is to generate a .csv file based on the target directory.
For the implementation ADC-Net, all files have the same name.
For example, image001.tiff will be the name for the first image in magnitude 1, magnitude 2, magnitude 3, ..., and compensated folders.

@author: dleninja
"""
#
import pandas as pd
from pathlib import Path
#
image_names = []
#
basepath =Path("dataset_dir/magnitude1")
files_in_basepath = basepath.iterdir()
#
for item in files_in_basepath:
	if item.is_file():
		image_names.append(item.name)
#
dictionary = {'image_names': image_names}
dataframe = pd.DataFrame(dictionary)
#
n_length = int(len(dataframe)*0.7)
dataframe_train = dataframe[:n_length]
dataframe_test = dataframe[n_length:]
#
dataframe_train.to_csv("train.csv", index=False)
dataframe_test.to_csv("test.csv", index=False)
