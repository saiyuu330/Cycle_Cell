import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms, datasets


def create_dataset(args):
    images = args.input
    labels = os.listdir(images)
    dataset_list = []

    trans = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    for label in labels:
        path = os.path.join(images, label)
        dataset = Dataset(root=path, transform=trans)
        dataset_list.append(dataset)

    return dataset_list


def create_dataloader(args):
    imageset = create_dataset(args)
    dataloader = DataLoader(imageset, batch_size=args.batch_size, shuffle=True)