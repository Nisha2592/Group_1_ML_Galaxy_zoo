import numpy as np
import os
import pandas as pd
from PIL import Image, ImageOps
from numpy import asarray
from dask import delayed



@delayed
def process_image_dask(image_path):
    try:
        img = Image.open(image_path)
        img_gray = ImageOps.grayscale(img)
        img_array = np.asarray(img_gray).flatten()
        filename = os.path.basename(image_path)
        return int(os.path.splitext(filename)[0]), img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def process_image_seq(image_dir):
    '''
    Sequential implementation. Loops through one image at a time. This is embarassingly 
    parallelizable. The task which here consists of 1. processing images, converting to grayscale 
    and flattening pixel values is CPU-bound, ie performance is determined promarily by how
    CPU can process it in contrast to I/O bound. 
    We can parallelize using Multiprocessing library or Dask. 
    '''

    # List to store image data 
    image_data = []
    image_names = []

    # Iterate over all files in the directory 
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.png')): #filter the image files
            image_path = os.path.join(image_dir, filename)

            # Open image and convert to grayscale
            img = Image.open(image_path)
            img_gray = ImageOps.grayscale(img)

            # Convert to a numpy array and flatter it to 1D
            img_array = np.asarray(img_gray).flatten()

            #store the image data and filename
            image_data.append(img_array)

            # Extract the base name without the extension
            image_name = int(os.path.splitext(filename)[0]) # Get only the root, ie w/o extension
    
            image_names.append(image_name)

    # convert to DataFrame
    image_data = pd.DataFrame(image_data)
    image_data.insert(0, "asset_id", image_names) # NOTE: asset_id values are object type. Need to convert to int64 before merging later.


    return image_data
