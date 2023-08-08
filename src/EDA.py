from typing import List
import os
import seaborn as sns
from easyimages import EasyImageList
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

def plot_counts(df: pd.DataFrame, tar_var: str) -> None:
    """
    Plot count distributions of a categorical variable in the DataFrame.

    This function generates two subplots: a pie chart and a count plot, to visualize the distribution
    of a categorical variable within the DataFrame. The pie chart displays the percentage distribution
    of each category, while the count plot provides a bar representation of category counts.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be visualized.
        tar_var (str): The target categorical variable column name.

    Returns:
        None
    """ 
    f,ax=plt.subplots(1,2,figsize=(18,8))
    df[tar_var].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
    ax[0].set_title(tar_var)
    ax[0].set_ylabel('')
    sns.countplot(data=df, x= df[tar_var],ax=ax[1])
    ax[1].set_title(tar_var)
    plt.show()

    return

def plot_img_size(df: pd.DataFrame, tar_var: str) -> None:
    """
    Plot a scatter plot of image sizes.

    This function extracts width and height information from the specified column of the DataFrame,
    and creates a scatter plot to visualize the relationship between image widths and heights.

    Args:
        df (pd.DataFrame): The DataFrame containing the image size data.
        tar_var (str): The target column containing image size information.

    Returns:
        None
    """
    # Extract the width and height columns from the DataFrame
    widths = df[tar_var].apply(lambda x: x[0])
    heights = df[tar_var].apply(lambda x: x[1])

    # Create a scatter plot of the image sizes
    plt.scatter(widths, heights)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Image Sizes")
    plt.tight_layout()
    plt.show()

    return

def aspect_ratio_plot(df: pd.DataFrame, tar_var: str, num_bins: int, range_min: float, range_max: float) -> None:
    """
    Plot a histogram of data values within a specified range.

    This function generates a histogram plot of the data values from the specified column of the DataFrame.
    The number of bins and the range of values are adjustable to customize the appearance of the histogram.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be visualized.
        tar_var (str): The target column containing the data values.
        num_bins (int): The number of bins for the histogram.
        range_min (float): The minimum value for the histogram range.
        range_max (float): The maximum value for the histogram range.

    Returns:
        None
    """
    # Create the histogram
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(df[tar_var], bins=num_bins, range=(range_min, range_max), density=False)

    # Add labels and title
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Data')

    # Show the plot
    plt.show()

def get_leaf_folders(path: str) -> List[str]:
    """
    Recursively get a list of all leaf folders (i.e. folders with no subfolders) in a directory and its subdirectories.

    Args:
        path: The path of the directory to search for leaf folders.

    Returns:
        A list of strings, where each string is the full path to a leaf folder in the specified directory or its subdirectories.
    """
    leaf_folders = set()
    for f in os.listdir(path):
        full_path = os.path.join(path, f)
        if os.path.isdir(full_path):
            if len(os.listdir(full_path)) == 0:
                # Ignore empty folders
                continue
            subfolders = get_leaf_folders(full_path)
            if len(subfolders) == 0:
                leaf_folders.add(full_path)
            else:
                leaf_folders.update(subfolders)
    return sorted(list(leaf_folders))


def create_EIL_from_directory(
    subfolder: str, sample: int = 500, size: int = 50
) -> EasyImageList:
    """
    Create an EasyImageList object from all image files in a directory and its subdirectories.

    Args:
        path: The path of the directory to search for image files.
        sample: The maximum number of images to include in the EasyImageList.
        size: The size of the thumbnails in the HTML visualization.

    Returns:
        An EasyImageList object created from all image files in the specified directory and its subdirectories.
    """
    # Create an EasyImageList object from the subfolder
    EIL = EasyImageList.from_folder(subfolder)
    EIL.symlink_images()
    EIL.html(sample=sample, size=size)
    return

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

def get_data_use(file_path: str) -> str:
    """
    Get the data usage type for an image file based on its file path.

    This function takes a file path as input and identifies the data usage type (train, test, or val) for the image
    based on certain keywords present in the file path. It is designed to work with a directory structure where
    different subdirectories represent different data splits (e.g., training, testing, validation).

    Args:
        file_path (str): The path of the image file.

    Returns:
        str or None: The data usage type for the image, represented as a string. The types are determined as follows:
            - If the file path contains the keyword "train", the data usage type returned is "train".
            - If the file path contains the keyword "test", the data usage type returned is "test".
            - If the file path contains the keyword "val", the data usage type returned is "val".
            - If none of the keywords are present in the file path, it returns None, indicating that the data usage
              type could not be determined for the given image file.
    """
    if "train" in file_path:
        return "train"
    elif "test" in file_path:
        return "test"
    elif "val" in file_path:
        return "val"
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


def canny_edge_plot(
    df: pd.DataFrame, label_list: list, label_col: str, file_path_col: str
) -> None:
    """
    Preprocesses images from a DataFrame containing file paths and labels, and displays 6 random images
    of each label category in a figure with 3 rows and 6 columns.

    Args:
        df (pd.DataFrame): DataFrame containing file paths and labels.
        label_col (str): Name of the column in `df` that contains the labels.
        file_path_col (str): Name of the column in `df` that contains the file paths.

    Returns:
        None
    """
    fig, axes = plt.subplots(
        nrows=3, ncols=6, figsize=(15, 10), subplot_kw={"xticks": [], "yticks": []}
    )

    for i, lbl in enumerate(label_list):
        img_paths = df.loc[df[label_col] == lbl, file_path_col].sample(
            n=6, random_state=1
        )

        for j, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (512, 512))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.Canny(img, 80, 100)
            axes[i][j].imshow(img)
            axes[i][j].set_title(lbl)

    fig.tight_layout()
    plt.show()

    return