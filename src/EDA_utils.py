import os
import pandas as pd
from PIL import Image
from .train import get_data_use

def prepare_df(path: str) -> pd.DataFrame:
    """
    Prepare a DataFrame with metadata for image files in a given directory.

    This function scans the directory and its subdirectories specified by 'path' and creates a DataFrame containing
    metadata for each image file. The DataFrame includes information such as file paths, labels (obtained using the
    'get_label' function), data usage (train, test, or val) obtained from 'get_data_use', image sizes (obtained from
    'get_image_size'), and the aspect ratio of each image.

    Args:
        path (str): The path to the directory containing image files.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the metadata for each image file, including file paths, labels,
        data usage, image sizes, and aspect ratios.
    """
    # Obtain a list of file paths for all image files in the directory and its subdirectories
    onlyfiles = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(path) for f in filenames]

    # Create a DataFrame with file paths
    df = pd.DataFrame({'file_path': onlyfiles})
    df['file_path'] = df['file_path'].astype(str)

    # Add label information based on keywords in file paths
    df["label"] = df["file_path"].apply(get_label)

    # Add data usage information based on keywords in file paths
    df["usage"] = df["file_path"].apply(get_data_use)

    # Add image sizes and aspect ratios using the 'get_image_size' function
    df["size"] = df["file_path"].apply(get_image_size)
    df["aspect_ratio"] = df["size"].apply(lambda x: x[0] / x[1])

    return df

def get_label(file_path: str) -> str:
    """
    Get the label for an image file based on its file path.

    This function takes a file path as input and identifies the label for the image based on certain keywords present
    in the file path. It is designed to work with a directory structure where different subdirectories represent
    different categories of images.

    Args:
        file_path (str): The path of the image file.

    Returns:
        str or None: The label for the image, represented as a string. The labels are determined as follows:
            - If the file path contains the keyword "bacteria", the label returned is "bacteria".
            - If the file path contains the keyword "virus", the label returned is "virus".
            - If the file path contains the keyword "normal", the label returned is "normal".
            - If none of the keywords are present in the file path, it returns None, indicating that the label could
              not be determined for the given image file.
    """
    if "bacteria" in file_path:
        return "bacteria"
    elif "virus" in file_path:
        return "virus"
    elif "normal" in file_path:
        return "normal"
    else:
        return None

def get_image_size(file_path: str) -> tuple:
    """
    Get the size (width and height) of an image file.

    This function takes a file path as input, opens the image using the Python Imaging Library (PIL), and returns the
    size of the image as a tuple (width, height).

    Args:
        file_path (str): The path of the image file.

    Returns:
        tuple: A tuple representing the size of the image in the format (width, height).
    """
    with Image.open(file_path) as img:
        return img.size