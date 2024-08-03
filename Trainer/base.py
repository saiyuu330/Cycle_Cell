import torch.optim as optim
import torch
import numpy as np

class Learner:
    def __init__(self, args):
        super().__init__()
        self.chk_point = args.resume_from
        self.start_epoch = 0
        self.epochs = args.epochs
        self.data_dir = args.input
        self.results_dir = args.output
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.seed = args.seed
        self.criterion = torch.nn.CrossEntropyLoss()
        self.data_loader = None


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
