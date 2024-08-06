import os
import argparse
from Trainer import *
import subprocess


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--mode", default="cycle_gan", type=str, help="cycle_gan or classification")
arg_parser.add_argument("--isTrain", default=True, type=bool, help="whether to train model")
arg_parser.add_argument("--input_dir", default="./image", type=str, help="path to input directory")
arg_parser.add_argument("--output_dir", default="./result", type=str, help="path to output directory")
arg_parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
arg_parser.add_argument("--batch_size", default=64, type=int, help="batch size")
arg_parser.add_argument("--learning_rate", default=1e-4, type=int, help="learning rate")
arg_parser.add_argument("--seed", default=42, type=int, help="random seed")
arg_parser.add_argument("--resume_from", default=None, type=int, help="resume from checkpoint")
arg_parser.add_argument("--device", default="cpu", type=str, help="use cpu or cuda")


def generate_model(arg):
    if arg.isTrain:
        trainer = CycleTrainer(arg)
        trainer.train()

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
    #result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #print(result.stdout)

    args = arg_parser.parse_args()
    if args.mode == "cycle_gan":
        generate_model(args)

    elif args.mode == "classification":
        classify(args)

    elif args.mode == "all":
        generate_model(args)
        classify(args)
