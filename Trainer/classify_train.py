import os
import torch
import torch.nn as nn
import torch.optim as optim
from .base import Learner, set_seed
from model.classifier import Classifier
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from data.dataset import create_class_dataset


class Trainer(Learner):
    def __init__(self, args):
        super().__init__(args)
        self.input_dir = args.input_dir
        self.dataset = create_class_dataset(self.input_dir, self.img_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = args.epochs
        self.seed = args.seed
        self.img_size = args.img_size

    def train(self):
        set_seed(self.seed)

        kfold = KFold(n_splits=5, shuffle=True, random_state=self.seed)

        all_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            print(f'FOLD {fold + 1}')
            print('--------------------------------')

            trainset = Subset(self.dataset, train_idx)
            valset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(valset, batch_size=self.batch_size, shuffle=False)

            model = Classifier(self.img_size, 6)
            model = model.to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            loss = 0.0
            for epoch in range(self.epochs):
                model.train()

                for batch in train_loader:
                    inputs, targets = batch

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, targets = batch
                        inputs = inputs.view(-1, 28 * 28)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

                accuracy = correct / total
                all_scores.append(accuracy)
                print(f'Fold {fold + 1} / Epoch {epoch +1} / loss: {loss.item()} / acc : {accuracy * 100:.2f}%')

        print('--------------------------------')
        print(f'Mean accuracy across all folds: {sum(all_scores) / len(all_scores) * 100:.2f}%')