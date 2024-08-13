import os
import torch
from model import Classifier
from utils import load_image


def predict(arg, image_path):
    classifier_path = f'./checkpoints/generator_A_to_B_{arg.epochs}.pth'

    out_dim = len(os.listdir(arg.input_dir))
    model = Classifier(arg.img_size, out_dim)

    classifier = Classifier(100, out_dim).to(arg.device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=arg.device))
    classifier.eval()

    image = load_image(image_path).to(arg.device)

    with torch.no_grad():
        output = classifier(image)

    pred = output.argmax(dim=1, keepdim=True)

    return [pred, output]