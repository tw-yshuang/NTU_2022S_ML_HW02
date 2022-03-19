import numpy as np
import torch
import torch.nn as nn
import src.net as net
from src.lr_schedule import get_cosine_schedule_with_warmup


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device State:', device)
    return device


class DL_Config(object):
    def __init__(self) -> None:
        self.basic_config()
        self.net_config()
        self.performance_config()
        self.save_config()

    def basic_config(self):
        self.SEED: int = 24
        self.NUM_EPOCH: int = 1000
        self.WARMUP_EPOCH: int = 10
        self.BATCH_SIZE: int = 64
        self.earlyStop: int or None = 200

        np.random.seed(self.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)

    def net_config(self, net_parameter: int = None, learning_rate: float = 1e-3):
        self.isClassified = True
        self.loss_func = nn.CrossEntropyLoss()

        if net_parameter is None:
            self.net = net.Classifier
            self.optimizer = torch.optim.AdamW
        else:
            self.net = self.net(net_parameter).to(get_device())
            self.optimizer = self.optimizer(self.net.parameters(), lr=learning_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, self.WARMUP_EPOCH, self.NUM_EPOCH)

    def performance_config(self):
        self.printPerformance: bool = True
        self.showPlot: bool = False
        self.savePerformance: bool = True
        self.savePlot: bool = True

    def save_config(self):
        self.saveDir = './out/'
        self.saveModel = True
        self.checkpoint = 50
        self.bestModelSave = True
        self.onlyParameters = True
