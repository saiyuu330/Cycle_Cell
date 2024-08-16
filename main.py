import os
from Trainer import *
import subprocess
import torch
from utils import draw_graph
from config import parser


def generate_model(arg):
    if arg.isTrain == "Train":
        trainer = CycleTrainer(arg)
        loss_list = trainer.train()
        draw_graph(loss_list)

    elif arg.isTrain == "Test":
        image_path = os.path.join(arg.test_dir + "/img01.png")
        cycle_predict(arg, image_path)


def classify(arg):
    if arg.isTrain:
        trainer = ClassifyTrainer(arg)
        trainer.train()
    else:
        image_path = os.path.join(arg.test_dir + "/generated_img01.png")
        label, properties = classify_predict(arg, image_path)

        print(label, properties)


if __name__ == "__main__":
    if torch.cuda.is_available():
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)

    args = parser()
    if args.mode == "cycle_gan":
        generate_model(args)

    elif args.mode == "classify":
        classify(args)

    elif args.mode == "all":
        generate_model(args)
        classify(args)
