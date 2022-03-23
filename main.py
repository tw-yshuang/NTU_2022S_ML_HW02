import csv
import threading
import torch
import numpy as np
from typing import List
from torch.utils.data.dataloader import DataLoader

import src.net as net
from submodules.FileTools.WordOperator import str_format
from src.data_process import get_dataloader
from src.train_process import DL_Model


class TaskThread(threading.Thread):
    def __init__(
        self,
        model: DL_Model,
        mode: str = 'train',
        loader: DataLoader or None = None,
        val_loader: DataLoader or None = None,
    ):
        threading.Thread.__init__(self)
        self.model = model
        self.mode = mode
        self.loader = loader
        self.val_loader = val_loader
        self.result = None

    def run(self):
        if self.mode == 'train':
            self.result = self.model.training(self.loader, self.val_loader)
        elif self.mode == 'test':
            self.result = self.model.testing(self.loader)

    def execute(self):
        self.start()
        try:
            self.join()
        except KeyboardInterrupt:
            self.model.earlyStop = self.model.epoch + 1


def train(model: DL_Model = None, loader: List[DataLoader] = None, data_dir: str = None, n_workers: int = 1, **kwargs):
    if model is None:
        model = DL_Model(**kwargs)

    if loader is not None:
        train_loader, valid_loader = loader
    else:
        train_loader, valid_loader, speaker_num = get_dataloader(batch_size=model.BATCH_SIZE, **kwargs['loader_config'])
        model.net_config(net_parameter=speaker_num, **kwargs['net_config'])

    task = TaskThread(model=model, loader=train_loader, val_loader=valid_loader)
    task.execute()


# def pre_train(model_path: str, loader: List[DataLoader] = None, data_dir: str = None, n_workers: int = 1, **kwargs):
def pre_train(model_path: str, loader: List[DataLoader] = None, **kwargs):
    model = DL_Model()
    model.net = getattr(net, model_path.split('/')[-2].split('_')[1])

    if loader is not None:
        train_loader, valid_loader = loader
    else:
        train_loader, valid_loader, speaker_num = get_dataloader(batch_size=model.BATCH_SIZE, **kwargs['loader_config'])

    model.net_config(net_parameter=speaker_num, **kwargs)
    model.load_model(model_path)

    task = TaskThread(model=model, loader=train_loader, val_loader=valid_loader)
    task.execute()


def full_train(model: DL_Model = None, loader: DataLoader = None, data_dir: str = None, n_workers: int = 1, **kwargs):
    if model is None:
        model = DL_Model()

    if loader is None:
        loader, speaker_num = get_dataloader(mode='full', batch_size=model.BATCH_SIZE, **kwargs['loader_config'])
        model.net_config(net_parameter=speaker_num, **kwargs)

    model.saveDir = './out/full/'
    task = TaskThread(model=model, loader=loader)
    task.execute()


def full_pre_train(
    model_path: str, model: DL_Model = None, loader: DataLoader = None, data_dir: str = None, n_workers: int = 1, **kwargs
):
    if model is None:
        model = DL_Model(**kwargs)

    model.load_model(model_path)

    if loader is None:
        loader, speaker_num = get_dataloader(mode='full', batch_size=model.BATCH_SIZE, **kwargs['loader_config'])
        model.net_config(net_parameter=speaker_num, **kwargs)

    task = TaskThread(model=model, loader=loader)
    task.execute()


def test(model_path: str, model: DL_Model = None, loader: List[DataLoader] = None, **kwargs):
    if model is None:
        model = DL_Model(**kwargs)

    if loader is None:
        loader, speaker_num = get_dataloader(mode='test', batch_size=model.BATCH_SIZE, **kwargs['loader_config'])
        model.net_config(net_parameter=speaker_num, **kwargs['net_config'])

    model.load_model(model_path)

    task = TaskThread(mode='test', model=model, loader=loader)
    task.execute()

    with open(f'{model_path[: model_path.rfind(".pickle")]}.csv', 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Id', 'Class'])
        for i, p in enumerate(task.result.astype(np.int32)):
            writer.writerow([i, p])

    print(str_format("done!!", fore='g'))


if __name__ == '__main__':
    loader_params = {
        'loader_config': {
            'data_dir': './Data/libriphone',
            'n_workers': 4,
        },
        # 'basic_confg': {
        # },
        'net_config': {
            'network': net.ClassifierInverseTriangle,
            'strcture': {
                'hidden_layers': 3,
                'max_hidden_dim': 512,
            },
            'optimizer': torch.optim.AdamW,
            'learning_rate': 1e-4,
            'lr_scheduler': None,
        },
        # 'performance_config': {},
        # 'save_config': {},
    }

    model = DL_Model(**loader_params)

    train(model=model, **loader_params)

    # pre_train(
    #     model_path='out/0320-1020_Classifier_CrossEntropyLoss_Adam-1.0e-03_BS-512/final_e092_2.377e-03.pickle',
    #     **loader_params,
    # )

    # full_train(**loader_params)
    # full_pre_train(**loader_params)

    # test(
    #     model_path='./out/0323-0418_ClassifierInverseTriangle_CrossEntropyLoss_AdamW-1.0e-04_BS-512/best-loss_e500_2.921e-03.pickle',
    #     model=model,
    #     **loader_params,
    # )
