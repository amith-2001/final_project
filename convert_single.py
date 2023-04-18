import os
import time
import pydicom
from PIL import Image
from PIL import ImageOps
import numpy as np
ds = pydicom.dcmread("C:/Users/amith/Downloads/final_project/DICOM/FLAIR/BRAINIX_DICOM_FLAIR_IM-0001-0002.dcm")
pixel_data = ds.pixel_array
image = Image.fromarray(pixel_data) or np.ones(np.shape(pixel_data))
# image.save("out.jpg")
# time.sleep(5)
print(image)
image.show()
# inverted_image = ImageOps.invert("out_000.jpg")
# inverted_image.save("inverted_image.jpg")
#

