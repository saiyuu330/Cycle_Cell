import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from base import Learner, set_seed
from model.classifier import Classifier


class Trainer(Learner):
    def __init__(self, args):
        super().__init__(args)
        self.input_dir = args.input_dir
        self.train_loader = DataLoader(...)  # Initialize your training data loader
        self.valid_loader = DataLoader(...)  # Initialize your validation data loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = args.epochs
        self.seed = args.seed
        self.lr = args.lr  # Learning rate

    def train(self):
        # Set the seed for reproducibility
        set_seed(self.seed)

        # Determine the output dimension
        out_dim = len(os.listdir(self.input_dir))
        model = Classifier(100, out_dim)
        model = model.to(self.device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for (x, y) in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                output = model(x)
                loss = criterion(output, y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_predictions += y.size(0)
                correct_predictions += (predicted == y).sum().item()

            # Calculate average loss and accuracy
            avg_loss = running_loss / len(self.train_loader)
            accuracy = correct_predictions / total_predictions * 100

            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        print("Training completed.")
