import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import Generator

# 가정: 학습된 Generator 모델이 저장된 경로
generator_path = 'generator_A_to_B.pth'

# Generator 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(input_nc=3, output_nc=3).to(device)
generator.load_state_dict(torch.load(generator_path, map_location=device))
generator.eval()  # 평가 모드로 전환

# 이미지 전처리 함수 정의
preprocess = transforms.Compose([
    transforms.Resize((100, 100)),  # 원본 이미지 크기와 동일하게 설정
    transforms.ToTensor(),  # 이미지 데이터를 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화
])


def load_image(img_path):
    """이미지를 로드하고 전처리합니다."""
    image = Image.open(img_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)  # 배치 차원 추가
    return image


def denormalize(tensor):
    """정규화된 텐서를 원래 이미지로 변환합니다."""
    tensor = tensor * 0.5 + 0.5  # [-1, 1] -> [0, 1]
    return tensor.clamp(0, 1)


def predict(image_path):
    """이미지를 변환하여 결과를 반환합니다."""
    image = load_image(image_path).to(device)

    # Generator A to B를 통해 이미지 변환
    with torch.no_grad():  # 역전파 비활성화
        fake_B = generator(image)

    # 변환된 이미지 후처리 및 시각화
    fake_B = denormalize(fake_B.squeeze().cpu())  # 배치 차원 제거 및 denormalize
    fake_B_image = transforms.ToPILImage()(fake_B)  # 텐서를 이미지로 변환

    return fake_B_image


# 예측할 이미지 경로
image_path = 'path_to_A_domain_image.jpg'

# 예측 수행 및 결과 시각화
fake_B_image = predict(image_path)
plt.imshow(fake_B_image)
plt.axis('off')
plt.show()
