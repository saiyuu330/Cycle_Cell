import os
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from PIL import Image
import random
import shutil
import numpy as np
import torch


class CustomImageDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # 이미지를 RGB 형식으로 변환
        if self.transform:
            image = self.transform(image)
        return image


def crop_randomly(image, image_size):
    """
    :param image: image to be cropped
    :param image_size: image size to be cropped
    :return: cropped image
    """

    width, height = image.size
    crop_width, crop_height = image_size, image_size

    x = random.randint(0, width - crop_width)
    y = random.randint(0, height - crop_height)

    return image.crop((x, y, x + crop_width, y + crop_height))


def splitting(directory, ratio=0.8):
    """
    :param ratio: split ratio
    :param directory: directory to split the dataset
    :return: split dataset
    """

    length = int(len(os.listdir(directory)) * ratio)
    lst_dir = os.listdir(directory)
    return lst_dir[:length], lst_dir[length:]


def check_augmentation_ratio(input_dir, num):
    count = 0
    sub_dir = os.listdir(input_dir)
    for s in sub_dir:
        folder = os.path.join(input_dir, s)
        count += len(os.listdir(folder))

    if count > num:
        return num / count
    else:
        return 0.8


def augment_dataset(input_dir, output_dir, num_crops, image_size):
    """
    :param input_dir: live cell images directory
    :param output_dir: ./03. cropped_images
    :param num_crops: number of data augmentations
    :param image_size: image size to be cropped
    :return: None; just save cropped images
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ratio = check_augmentation_ratio(input_dir, num_crops)
    counter = 0

    for directory in os.listdir(input_dir):
        folder = os.path.join(input_dir, directory)
        split_files_1, split_files_2 = splitting(folder, ratio)

        while counter < num_crops:
            for file in split_files_1:
                img = os.path.join(folder, file)
                image = Image.open(img).convert("RGB")
                cropped_image = crop_randomly(image, image_size)
                filename = os.path.join(output_dir, file[:-4] + "_" + str(counter) + ".png")
                cropped_image.save(filename)
                counter += 1
                if counter == num_crops:
                    break

        """
        ///////////////////////////////////
        ///                             ///
        ///    make test images set     ///
        ///                             ///
        ///////////////////////////////////
        """

        new_dir = "./image/04. test_images"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        new_folder = os.path.join(new_dir, directory)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        for file in split_files_2:
            source_file = os.path.join(folder, file)
            destination_file = os.path.join(new_folder, file)
            shutil.copy2(source_file, destination_file)


def create_cycle_dataset(path, image_size):
    """
    :param path: directory to make dataset
    :param image_size: resize image size
    :return: dataset
    """

    trans = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CustomImageDataset(path, trans)
    return dataset


def create_cycle_loader(in_dir, b_size, mode, image_size):
    """
    :param in_dir: directory to make dataloader
    :param b_size: batch size
    :param mode: set shuffle
    :param image_size: for use dataset
    :return: dataloader
    """

    imageset = create_cycle_dataset(in_dir, image_size)

    is_train = False
    if mode == "Train":
        is_train = True

    dataloader = DataLoader(imageset, batch_size=b_size, shuffle=is_train)
    return dataloader


def create_class_dataset(path, image_size):
    images = []
    labels = []
    class_names = ['oligomycin', 'control', 'rotenone', 'untreated', '3xSNCA', 'oligomer']
    num_classes = len(class_names)
    folder = os.path.join(path, "02. target data")

    for filename in os.listdir(folder):
        label = -1
        for i, class_name in enumerate(class_names):
            if class_name in filename:
                label = i
                break

        if label == -1:
            continue

        file_path = os.path.join(folder, filename)
        file = Image.open(file_path)

        trans = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        arr_file = trans(file)

        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[label] = 1.0

        images.append(arr_file)
        labels.append(one_hot)

    arr_label = np.array(labels)
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(arr_label, dtype=torch.float32)

    dataset = TensorDataset(images_tensor, labels_tensor)
    return dataset
