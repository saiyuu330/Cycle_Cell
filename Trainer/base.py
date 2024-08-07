import torch.optim as optim
import torch
import numpy as np


class Learner:
    def __init__(self, args):
        super().__init__()
        self.chk_point = args.resume_from
        self.start_epoch = args.start_epoch
        self.epochs = args.epochs
        self.data_dir = args.input_dir
        self.results_dir = args.output_dir
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.seed = args.seed
        self.criterion = torch.nn.CrossEntropyLoss()
        self.is_train = args.isTrain
        self.img_size = args.img_size


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
