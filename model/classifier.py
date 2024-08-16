import torch
import torch.nn as nn

class Classifier(nn.Module):
    """
    classification model
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),

            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.MaxPool2d(kernel_size=8, stride=8)
        )
        self.fc = nn.Linear(in_features=64*64, out_features=output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
