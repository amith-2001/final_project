import os
import pydicom
from PIL import Image
from PIL import ImageOps
# Define the input and output directories
input_dir = 'C:/Users/amith/Downloads/final_project/DICOM/FLAIR'
output_dir = 'C:/Users/amith/Downloads/final_project/out'

# Loop through all DICOM files in the input directory
for filename in os.listdir(input_dir):
    if "dcm" in filename:

        # Load the DICOM file using pydicom
        ds = pydicom.dcmread(os.path.join(input_dir, filename))
        print(filename)
        # Extract the pixel data and convert to a Pillow Image object
        pixel_data = ds.pixel_array
        image = Image.fromarray(pixel_data)

        # image = ImageOps.invert(image)
        print(image)
        # Save the image as a JPG file in the output directory
        output_filename = filename + '.jpg'
        image.show(os.path.join(output_dir, filename))