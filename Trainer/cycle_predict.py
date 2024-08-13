import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from model import Generator
from utils import load_image, denormalize


def predict(arg, image_path):
    generator_path = f'./checkpoints/generator_A_to_B_{arg.epochs}.pth'

    generator = Generator(input_nc=3, output_nc=3).to(arg.device)
    generator.load_state_dict(torch.load(generator_path, map_location=arg.device, weights_only=True)['generator_G_state_dict'])

    generator.eval()

    image = load_image(image_path, arg.img_size).to(arg.device)
    print(f"Input image shape: {image.shape}")  # 입력 이미지 크기 확인

    with torch.no_grad():
        fake_b = generator(image)
    print(f"Output image shape: {fake_b.shape}")  # 출력 이미지 크기 확인

    fake_b = denormalize(fake_b.squeeze(0).cpu())
    print(f"Denormalized image shape: {fake_b.shape}")  # 디노멀라이즈 후 크기 확인
    fake_b_image = transforms.ToPILImage()(fake_b)
    print(f"Image type: {type(fake_b_image)}")  # 이미지 타입 확인

    save_path = f'./test_data/generated_image1_{arg.epochs}.png'
    fake_b_image.save(save_path)

    plt.imshow(fake_b_image)
    plt.axis('off')
    plt.show()
