import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.multiclass import unique_labels
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image as PilImage

import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from tensorboard import program


def get_label(file_path: str):
    if "bacteria" in file_path:
        return "1"
    elif "virus" in file_path:
        return "2"
    elif "normal" in file_path:
        return "0"
    else:
        return None


def get_data_use(file_path: str):
    if "train" in file_path:
        return "train"
    elif "test" in file_path:
        return "test"
    elif "val" in file_path:
        return "val"
    else:
        return None


def setup_data(path: str) -> pd.DataFrame:
    onlyfiles = [
        os.path.join(dirpath, f)
        for (dirpath, dirnames, filenames) in os.walk(path)
        for f in filenames
    ]
    df = pd.DataFrame({"file_path": onlyfiles})

    df["label"] = df["file_path"].apply(get_label)
    df["usage"] = df["file_path"].apply(get_data_use)

    train_df = df[df["usage"] == "train"]
    test_df = df[df["usage"] == "test"]
    val_df = df[df["usage"] == "val"]

    return train_df, test_df, val_df


def train_prepare(mode: str = "train"):
    train_df, test_df, val_df = setup_data(
        path="/home/justin/Desktop/Resume/Github/Explainable-AI/data"
    )

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


class ImageClassificationModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features

        # Add more layers
        self.layers = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.output_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("val_loss", loss)
        preds = torch.argmax(y_hat, dim=1)
        acc = torch.sum(preds == y) / float(y.shape[0])
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        # get the true labels and predicted labels
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        true_labels.append(y.cpu().numpy())
        pred_labels.append(preds.cpu().numpy())

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)

        # Define the learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",  # Metric to monitor for reducing LR
                "interval": "epoch",  # Adjust LR after each epoch
                "frequency": 1,  # Apply LR scheduler every epoch
            },
        }


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


def cm(true_labels, pred_labels):
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


def f1(true_labels, pred_labels):
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")
    f1 = f1_score(true_labels, pred_labels, average="weighted")

    # Print the scores
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def run() -> None:
    trainer, model, train_loader, val_loader, test_loader = train_prepare(mode="train")

    # train the model
    trainer.fit(model, train_loader, val_loader)
    model.eval()

    # evaluate on test set
    trainer.test(model, test_loader)
    

    true_label = np.concatenate(true_labels)
    pred_label = np.concatenate(pred_labels)

    cm(true_label, pred_label)

    f1(true_label, pred_label)

    torch.save(model.state_dict(), "src/model_assets/model.pt")

    return


if __name__ == "__main__":
    true_labels = []
    pred_labels = []
    run()
    os.system('tensorboard --logdir="tb_logs" --host=127.0.0.1')