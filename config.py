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
    def __init__(
        self,
        basic_confg: dict = {},
        net_config: dict = {},
        performance_config: dict = {},
        save_config: dict = {},
        **kwargs,
    ) -> None:
        self.basic_config(**basic_confg)
        self.net_config(**net_config)
        self.performance_config(**performance_config)
        self.save_config(**save_config)

        self.extraHyperConfig = {
            **kwargs,
            'basic_confg': basic_confg,
            'net_config': net_config,
            'performance_config': performance_config,
            'save_config': save_config,
        }
        print(self.extraHyperConfig)

    def basic_config(
        self,
        SEED: int = 42,
        NUM_EPOCH: int = 500,
        WARMUP_EPOCH: int = 10,
        BATCH_SIZE: int = 512,
        earlyStop: int or None = 50,
        **kwargs,
    ):
        self.SEED = SEED
        self.NUM_EPOCH = NUM_EPOCH
        self.WARMUP_EPOCH = WARMUP_EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.earlyStop = earlyStop

        np.random.seed(self.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)

    def net_config(
        self,
        net_parameter: int = None,
        network: nn.Module = net.Classifier,
        loss_func: nn = nn.CrossEntropyLoss(),
        optimizer: torch.optim = torch.optim.AdamW,
        learning_rate: float = 1e-4,
        lr_scheduler: torch.optim.lr_scheduler = False,
        **kwargs,
    ):
        self.isClassified = True
        self.loss_func = loss_func

        if net_parameter is None:
            self.net = network
            self.optimizer = optimizer
            self.learning_rate = learning_rate
        else:
            self.net = self.net(net_parameter, **kwargs['strcture']).to(get_device())
            self.optimizer = self.optimizer(self.net.parameters(), lr=self.learning_rate)
            self.lr_scheduler = (
                lr_scheduler
                if lr_scheduler is not False
                else get_cosine_schedule_with_warmup(self.optimizer, self.WARMUP_EPOCH, self.NUM_EPOCH)
            )
            # self.lr_scheduler = lr_scheedrduler if lr_scheduler is not False else None

    def performance_config(
        self,
        printPerformance: bool = True,
        showPlot: bool = False,
        savePerformance: bool = True,
        savePlot: bool = True,
        **kwargs,
    ):
        self.printPerformance = printPerformance
        self.showPlot = showPlot
        self.savePerformance = savePerformance
        self.savePlot = savePlot

    def save_config(
        self,
        saveDir='./out/',
        saveModel=True,
        checkpoint=50,
        bestModelSave=True,
        onlyParameters=True,
        **kwargs,
    ):
        self.saveDir = saveDir
        self.saveModel = saveModel
        self.checkpoint = checkpoint
        self.bestModelSave = bestModelSave
        self.onlyParameters = onlyParameters
