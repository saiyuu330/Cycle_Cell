import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


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


def create_dataset(path):
    trans = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CustomImageDataset(path, trans)
    return dataset


def create_dataloader(in_dir, b_size, is_train):
    imageset = create_dataset(in_dir)
    dataloader = DataLoader(imageset, batch_size=b_size, shuffle=is_train, num_workers=4)
    return dataloader
