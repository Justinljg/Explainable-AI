from typing import List
import os
import seaborn as sns
from easyimages import EasyImageList
import cv2
import matplotlib.pyplot as plt
import pandas as pd


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


def plot_counts(df: pd.DataFrame, tar_var: str) -> None:
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    df[tar_var].value_counts().plot.pie(autopct="%1.1f%%", ax=ax[0], shadow=True)
    ax[0].set_title(tar_var)
    ax[0].set_ylabel("")
    sns.countplot(data=df, x=df[tar_var], ax=ax[1])
    ax[1].set_title(tar_var)
    plt.show()

    return


def plot_img_size(df: pd.DataFrame, tar_var: str) -> None:
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


def aspect_ratio_plot(
    df: pd.DataFrame, tar_var: str, num_bins: int, range_min: float, range_max: float
) -> None:
    # Create the histogram
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(
        df[tar_var], bins=num_bins, range=(range_min, range_max), density=False
    )

    # Add labels and title
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Data")

    # Show the plot
    plt.show()


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
