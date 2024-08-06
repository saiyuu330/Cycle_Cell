from PIL import Image
from matplotlib import pyplot as plt
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


def draw_graph(loss):
    plt.figure(figsize=(5, 15))
    loss_arr = loss.data.numpy()
    loss_arr = loss_arr.T

    loss_D_A = loss_arr[0]
    loss_D_B = loss_arr[1]
    loss_G = loss_arr[2]

    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(range(100), loss_D_A)

    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(range(100), loss_D_B)

    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(range(100), loss_G)

    plt.show()
    plt.close()