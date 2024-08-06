import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random


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
    width, height = image.size
    crop_width, crop_height = image_size, image_size

    x = random.randint(0, width - crop_width)
    y = random.randint(0, height - crop_height)

    return image.crop((x, y, x + crop_width, y + crop_height))


def augment_dataset(input_dir, output_dir, num_crops, image_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]

    counter = 0
    while counter < num_crops:
        for img_name in images:
            img_path = os.path.join(input_dir, img_name)
            image = Image.open(img_path)
            cropped_image = crop_randomly(image, image_size)
            output_path = os.path.join(output_dir, f'{img_name[:-5]}_cropped_{counter}.png')
            cropped_image.save(output_path)
            counter += 1
            if counter >= num_crops:
                break


def create_dataset(path, image_size):
    trans = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CustomImageDataset(path, trans)
    return dataset


def create_dataloader(in_dir, b_size, is_train, image_size):
    imageset = create_dataset(in_dir, image_size)
    dataloader = DataLoader(imageset, batch_size=b_size, shuffle=is_train)
    return dataloader
