import numpy as np
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
            for epoch in range(self.epochs):
                model.train()
                train_acc = []
                train_loss = []
                for batch in train_loader:
                    inputs, targets = batch

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    acc = (outputs.argmax(dim=-1) == targets.to(self.device).argmax(dim=-1)).float().mean()
                    train_loss.append(loss.item())
                    train_acc.append(acc)
                TA = sum(train_acc) / len(train_acc)
                TL = sum(train_loss) / len(train_loss)
                model.eval()
                loss = 0.0
                valid_acc = []
                valid_loss = []
                for batch in val_loader:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)

                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = criterion(outputs.cpu(), targets.to(float))

                    acc = (outputs.argmax(dim=-1) == targets.to(self.device).argmax(dim=-1)).float().mean()
                    valid_loss.append(loss.item())
                    valid_acc.append(acc)
                VA = sum(valid_acc) / len(valid_acc)
                VL = sum(valid_loss) / len(valid_loss)
                print(f'Fold {fold + 1} / Epoch {epoch +1} / train loss: {TA} / train acc: {TA}% / valid loss: {VL} / valid acc : {VA}%')

        print('--------------------------------')