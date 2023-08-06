import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.multiclass import unique_labels

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image as PilImage

import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from model import ImageClassificationModel

def get_label_onehot(file_path: str) -> str or None:
    """
    Get the label for an image file based on its file path.

    This function takes a file path as input and identifies the label for the image based on certain keywords
    present in the file path. It is designed to work with a directory structure where different subdirectories
    represent different categories of images.

    Args:
        file_path (str): The path of the image file.

    Returns:
        str or None: The label for the image, represented as a string. The labels are determined as follows:
            - If the file path contains the keyword "bacteria", the label returned is "1".
            - If the file path contains the keyword "virus", the label returned is "2".
            - If the file path contains the keyword "normal", the label returned is "0".
            - If none of the keywords are present in the file path, it returns None, indicating that the label
              could not be determined for the given image file.
    """
    if "bacteria" in file_path:
        return "1"
    elif "virus" in file_path:
        return "2"
    elif "normal" in file_path:
        return "0"
    else:
        return None
    
def get_data_use(file_path: str) -> str or None:
    """
    Determine the data usage type based on the file path.

    This function takes a file path as input and identifies the data usage type based on certain keywords
    present in the file path. It is designed to work with a directory structure where different subdirectories
    represent different sets of data, such as training, testing, or validation datasets.

    Args:
        file_path (str): The path of the file.

    Returns:
        str or None: The data usage type, represented as a string. The function determines the data usage
        based on the following keywords present in the file path:
            - If the file path contains the keyword "train", the function returns "train".
            - If the file path contains the keyword "test", the function returns "test".
            - If the file path contains the keyword "val", the function returns "val".
            - If none of the keywords are present in the file path, it returns None, indicating that the
              data usage type could not be determined for the given file.]
    """
    if "train" in file_path:
        return "train"
    elif "test" in file_path:
        return "test"
    elif "val" in file_path:
        return "val"
    else:
        return None

def setup_data(path: str) -> pd.DataFrame:
    """
    Set up a DataFrame with file paths, labels, and data usage types for images in a directory and its subdirectories.

    This function recursively searches for image files in the specified directory and its subdirectories.
    For each image file found, it creates a pandas DataFrame containing three columns: 'file_path', 'label', and 'usage'.
    The 'file_path' column stores the absolute path of each image file. The 'label' column is populated based on
    the 'get_label_onehot' function, which determines the label of the image based on the file path. The 'usage' column
    is determined using the 'get_data_use' function, which identifies the data usage type of the image (e.g., train,
    test, or validation) based on the file path.

    Args:
        path (str): The path of the directory to search for image files.

    Returns:
        tuple: A tuple of pandas DataFrames representing the datasets with different data usages.
            - The first DataFrame corresponds to the 'train' dataset.
            - The second DataFrame corresponds to the 'test' dataset.
            - The third DataFrame corresponds to the 'val' (validation) dataset.
    """
    onlyfiles = [
        os.path.join(dirpath, f)
        for (dirpath, dirnames, filenames) in os.walk(path)
        for f in filenames
    ]
    df = pd.DataFrame({"file_path": onlyfiles})

    df["label"] = df["file_path"].apply(get_label_onehot)
    df["usage"] = df["file_path"].apply(get_data_use)

    train_df = df[df["usage"] == "train"]
    test_df = df[df["usage"] == "test"]
    val_df = df[df["usage"] == "val"]

    return train_df, test_df, val_df


def train_prepare(mode: str = "train") -> tuple:
    """
    Prepare data, model, and trainer objects for image classification training or fine-tuning.

    This function sets up the necessary components for training or fine-tuning an image classification model.
    It loads the data using the 'setup_data' function, initializes custom datasets, creates an image classification
    model using 'ImageClassificationModel', sets up data loaders for train, validation, and test sets,
    and initializes a PyTorch Lightning trainer with specified configurations.

    Args:
        mode (str, optional): The mode to prepare for. Can be either "train" or "finetune".
            Default is "train".

    Returns:
        tuple: A tuple containing the following objects:
            - trainer (pytorch_lightning.Trainer): The PyTorch Lightning trainer object.
            - model (ImageClassificationModel): The image classification model.
            - train_loader (DataLoader): Data loader for the training dataset.
            - val_loader (DataLoader): Data loader for the validation dataset.
            - test_loader (DataLoader): Data loader for the test dataset.
    """
    train_df, test_df, val_df = setup_data(path="data")

    # set up transforms for your images
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # initialize custom datasets
    train_dataset = CustomDataset(train_df, transform)
    val_dataset = CustomDataset(val_df, transform)
    test_dataset = CustomDataset(test_df, transform)

    # initialize the model
    model = ImageClassificationModel(num_classes=3)

    # set up data loaders for your train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=3)

    # set up a logger for TensorBoard
    logger = TensorBoardLogger("tb_logs", name="image_classification")

    trainer = pl.Trainer(
        max_epochs=17,
        logger=logger,
        accelerator="auto",
        devices="auto",
        strategy="auto",
    )

    mode = "train"

    if mode == "train":
        for param in model.mobilenet.parameters():
            param.requires_grad = True

        for param in model.layers.parameters():
            param.requires_grad = True

        for param in model.output_layer.parameters():
            param.requires_grad = True

    if mode == "finetune":
        for param in model.mobilenet.parameters():
            param.requires_grad = False

        for param in model.layers.parameters():
            param.requires_grad = True

        for param in model.output_layer.parameters():
            param.requires_grad = True

    return trainer, model, train_loader, val_loader, test_loader

class CustomDataset(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.iloc[index]["file_path"]

        # open the image and convert it to RGB
        img = PilImage.open(img_path).convert("RGB")

        # resize the image to 224x224
        img = img.resize((224, 224))

        # apply the specified transforms
        img = self.transforms(img)

        label = self.data.iloc[index]["label"]
        # convert the label to an integer and then to a tensor
        label = torch.tensor(int(label))
        return img, label


def cm(true_labels, pred_labels) -> None:
    """
    Plot the confusion matrix for evaluating the performance of a classification model.

    This function takes true labels and predicted labels as inputs and creates a confusion matrix to visualize
    the model's performance on a classification task. It uses matplotlib to generate a heatmap with color-coded
    values representing the number of true positives, false positives, true negatives, and false negatives for
    each class.

    Args:
        true_labels (array-like): The true labels of the data.
        pred_labels (array-like): The predicted labels of the data.

    Returns:
        None: The function displays the confusion matrix as a heatmap using matplotlib and does not return anything.
    """
    # Create the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Get the unique class labels
    classes = unique_labels(true_labels, pred_labels)

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the x-axis labels if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over the data and create text annotations
    fmt = ".2f"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # Show the plot
    plt.tight_layout()
    plt.show()

    return


def f1(true_labels, pred_labels) -> None:
    """
    Calculate and print the precision, recall, and F1 score for a classification model's performance evaluation.

    This function takes the true labels and predicted labels of a classification task and calculates the precision,
    recall, and F1 score using scikit-learn's precision_score, recall_score, and f1_score functions. These metrics
    are common evaluation measures used to assess the performance of a classification model.

    Precision measures the proportion of true positive predictions out of all positive predictions. It is useful when
    the cost of false positives is high, and we want to minimize the number of false alarms.

    Recall (also called sensitivity or true positive rate) measures the proportion of true positive predictions out of
    all actual positive instances in the dataset. It is useful when the cost of false negatives is high, and we want to
    minimize the number of missed positive instances.

    F1 score is the harmonic mean of precision and recall. It is a balance between precision and recall and is useful
    when we want to consider both false positives and false negatives in the evaluation.

    Args:
        true_labels (array-like): The true labels of the data.
        pred_labels (array-like): The predicted labels generated by a classification model.

    Returns:
        None: The function prints the calculated precision, recall, and F1 score to the console and does not return
        any value.
    """
    # Calculate precision, recall, and F1 score using weighted averaging
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")
    f1 = f1_score(true_labels, pred_labels, average="weighted")

    # Print the scores
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return


def run(modes:str) -> None:
    """
    Run the complete pipeline for training, evaluating, and saving a machine learning model.

    This function orchestrates the entire machine learning pipeline, including data preparation, model training,
    evaluation on test data, and saving the trained model. It assumes the existence of a custom Trainer class for
    training the model, a custom Model class representing the machine learning model, and a custom data loading
    function named "train_prepare" that prepares the data for training.

    The function first prepares the necessary components for training by calling the "train_prepare" function, which
    returns a Trainer object, the Model object, and the data loaders for training, validation, and testing.

    Then, the model is trained using the "trainer.fit" method, with the training and validation data from the respective
    data loaders. After training, the model is switched to evaluation mode using "model.eval()" to disable dropout and
    other regularization layers.

    Next, the trained model is evaluated on the test set using the "trainer.test" method with the test data loader.

    The function then concatenates the true labels and predicted labels from the test set into numpy arrays to prepare
    them for further analysis.

    The confusion matrix is plotted using the "cm" function, which provides a visual representation of the model's
    classification performance on the test data.

    The precision, recall, and F1 score are calculated and printed using the "f1" function, providing additional
    evaluation metrics for the model's performance.

    Finally, the trained model's state dictionary is saved to a file named "model.pt" in the "model_assets/"
    directory.

    Returns:
        None: The function does not return anything.
    """
    # Prepare the Trainer, Model, and data loaders for training
    trainer, model, train_loader, val_loader, test_loader = train_prepare(mode=modes)

    # Train the model
    trainer.fit(model, train_loader, val_loader)
    model.eval()

    # Evaluate the model on the test set
    trainer.test(model, test_loader)

    # Concatenate true labels and predicted labels from the test set
    true_label = np.concatenate(true_labels)
    pred_label = np.concatenate(pred_labels)

    # Plot the confusion matrix for evaluation
    cm(true_label, pred_label)

    # Calculate and print precision, recall, and F1 score for evaluation
    f1(true_label, pred_label)

    # Save the trained model's state dictionary to a file
    torch.save(model.state_dict(), "model_assets/model.pt")

    return


if __name__ == "__main__":
    true_labels = []
    pred_labels = []
    run(modes="train")
    os.system('tensorboard --logdir="tb_logs" --host=127.0.0.1')
