
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

import lightning as pl

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
    
true_labels=[]
pred_labels=[]