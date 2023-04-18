from tqdm import tqdm
import os
from random import randint

import numpy as np
import pandas as pd

import nibabel as nib
import pydicom as pdm
import nilearn as nl
import nilearn.plotting as nlplt

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim

import imageio
from skimage.transform import resize
from skimage.util import montage

from IPython.display import Image as show_gif
import warnings
warnings.simplefilter("ignore")


sample_filename = 'C:/Users/amith/Downloads/final_project/input/BraTS20_Training_001/BraTS20_Training_001_flair.nii'
sample_filename_mask = 'C:/Users/amith/Downloads/final_project/input/BraTS20_Training_001/BraTS20_Training_001_seg.nii'

sample_img = nib.load(sample_filename)
sample_img = np.asanyarray(sample_img.dataobj)
sample_mask = nib.load(sample_filename_mask)
sample_mask = np.asanyarray(sample_mask.dataobj)
print("img shape ->", sample_img.shape)
print("mask shape ->", sample_mask.shape)


slice_n = 100
fig, ax = plt.subplots(2, 3, figsize=(25, 15))

ax[0, 0].imshow(sample_img[slice_n, :, :])
ax[0, 0].set_title(f"image slice number {slice_n} along the x-axis", fontsize=18, color="red")
ax[1, 0].imshow(sample_mask[slice_n, :, :])
ax[1, 0].set_title(f"mask slice {slice_n} along the x-axis", fontsize=18, color="red")

ax[0, 1].imshow(sample_img[:, slice_n, :])
ax[0, 1].set_title(f"image slice number {slice_n} along the y-axis", fontsize=18, color="red")
ax[1, 1].imshow(sample_mask[:, slice_n, :])
ax[1, 1].set_title(f"mask slice number {slice_n} along the y-axis", fontsize=18, color="red")

ax[0, 2].imshow(sample_img[:, :, slice_n])
ax[0, 2].set_title(f"image slice number {slice_n} along the z-axis", fontsize=18, color="red")
ax[1, 2].imshow(sample_mask[:, :, slice_n])
ax[1, 2].set_title(f"mask slice number {slice_n}along the z-axis", fontsize=18, color="red")
# fig.tight_layout()
# plt.show()



title = sample_filename.replace(".", "/").split("/")[-2]
filename = title+"_3d.gif"

data_to_3dgif = Image3dToGIF3d()#img_dim = (120, 120, 78)
transformed_data = data_to_3dgif.get_transformed_data(sample_img)
data_to_3dgif.plot_cube(
    transformed_data[:38, :47, :35],#[:77, :105, :55]
    title=title,
    make_gif=True,
    path_to_save=filename
)
show_gif(filename, format='png')