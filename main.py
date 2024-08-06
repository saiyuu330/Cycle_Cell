import os
from Trainer import *
import subprocess
import torch
from utils import draw_graph
from config import parser


def generate_model(arg):
    if arg.isTrain:
        trainer = CycleTrainer(arg)
        return trainer.train()

    else:
        cycle_predict(arg, 'test_data/image01.jpg')


def classify(arg):
    if arg.isTrain:
        trainer = ClassifyTrainer(arg)
        trainer.train()
    else:
        label, properties = classify_predict('test_data/image01.jpg')
        list_label = os.listdir(arg.input_dir)

        print(list_label[label], properties)


if __name__ == "__main__":
    if torch.cuda.is_available():
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)

    args = parser()
    if args.mode == "cycle_gan":
        loss_list = generate_model(args)
        draw_graph(loss_list)

    elif args.mode == "classification":
        classify(args)

    elif args.mode == "all":
        generate_model(args)
        classify(args)
