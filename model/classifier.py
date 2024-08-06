import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    classification model
    """
    def __init__(self, input_size, output_size):
        super().__init__(self)
        self.conv1 = torch.Sequential(
            torch.nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )
        self.fc = torch.Linear(in_features=256, out_features=output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return torch.softmax(x, dim=1)

