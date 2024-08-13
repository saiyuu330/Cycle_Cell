import argparse


def parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", default="cycle_gan", type=str, help="cycle_gan or classification")
    arg_parser.add_argument("--isTrain", default="Train", type=str, help="whether to train model")
    arg_parser.add_argument("--input_dir", default="./image", type=str, help="path to input directory")
    arg_parser.add_argument("--output_dir", default="./result", type=str, help="path to output directory")
    arg_parser.add_argument("--test_dir", default="./test_data", type=str, help="path to test directory")
    arg_parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    arg_parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    arg_parser.add_argument("--learning_rate", default=1e-4, type=int, help="learning rate")
    arg_parser.add_argument("--seed", default=42, type=int, help="random seed")
    arg_parser.add_argument("--resume_from", default=None, type=int, help="resume from checkpoint")
    arg_parser.add_argument("--device", default="cpu", type=str, help="use cpu or cuda")
    arg_parser.add_argument("--img_size", default=256, type=int, help="image size")
    arg_parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")

    return arg_parser.parse_args()