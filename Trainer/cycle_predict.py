import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from model import Generator
from utils import load_image, denormalize


def predict(arg, image_path):
    generator_path = './checkpoints/generator_A_to_B_100.pth'

    generator = Generator(input_nc=3, output_nc=3).to(arg.device)
    generator.load_state_dict(torch.load(generator_path, map_location=arg.device))
    generator.eval()

    image = load_image(image_path).to(arg.device)

    with torch.no_grad():
        fake_b= generator(image)

    fake_b = denormalize(fake_b.squeeze().cpu())
    fake_b_image = transforms.ToPILImage()(fake_b)

    plt.imshow(fake_b_image)
    plt.axis('off')
    plt.show()
    plt.close()
