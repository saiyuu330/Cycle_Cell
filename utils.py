from PIL import Image
from torchvision import transforms


def load_image(img_path):
    image = Image.open(img_path).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = preprocess(image).unsqueeze(0)
    return image


def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5  # [-1, 1] -> [0, 1]
    return tensor.clamp(0, 1)