import sys, os, glob, time, json
from typing import List
import numpy as np
import torch
from torch.autograd.grad_mode import no_grad
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(__package__))
from config import DL_Config, get_device
from submodules.ML_Tools.ModelPerform import ModelPerform
from submodules.FileTools.WordOperator import str_format
from submodules.FileTools.PickleOperator import save_pickle, load_pickle


class DL_Performance(object):
    def __init__(self) -> None:
        self.train_loss_ls: List[float] = []
        self.val_loss_ls: List[float] = []
        self.train_acc_ls: List[float] = []
        self.val_acc_ls: List[float] = []

        self.best_acc_epoch: int = 0
        self.best_loss_epoch: int = 0

    def print(self, epoch: int, total_epoch: int, time_start: time, end: str = '\n'):
        if len(self.train_loss_ls) <= epoch:
            raise ProcessLookupError(str_format("Wrong no. of epoch !!", style='blink', fore='r'))

        val_word = " | "
        if len(self.val_loss_ls) > epoch:  # avoid error that doesn't have validation in pre-train process.
            if len(self.val_acc_ls) > epoch:
                val_acc = (
                    str_format(f'{self.val_acc_ls[epoch]*100:.2f}', fore='y')
                    if self.val_acc_ls[self.best_acc_epoch] == self.val_acc_ls[epoch]
                    else f'{self.val_acc_ls[epoch]*100:.2f}'
                )
                val_word += f"Val acc: {val_acc}, "

            val_loss = (
                str_format(f'{self.val_loss_ls[epoch]:.5e}', fore='y')
                if self.val_loss_ls[self.best_loss_epoch] == self.val_loss_ls[epoch]
                else f'{self.val_loss_ls[epoch]:.5e}'
            )
            val_word += f"loss: {val_loss}"

        train_word = f"Train acc: {self.train_acc_ls[epoch]*100:.2f}, " if len(self.train_acc_ls) > epoch else "Train "
        train_word += f"loss: {self.train_loss_ls[epoch]:.5e}"

        print(f"[{epoch+1:>2d}/{total_epoch}] {time.time() - time_start:.3f}sec, {train_word}{val_word}", end=end)

    def visualize_info(self, showPlot: bool = False, savePlot: bool = True, saveCSV: bool = True, saveDir: str or None = './out'):
        self.visualize = ModelPerform(
            self.train_loss_ls,
            self.val_loss_ls if self.val_loss_ls != [] else None,
            self.train_acc_ls if self.train_acc_ls != [] else None,
            self.val_acc_ls if self.val_acc_ls != [] else None,
            saveDir,
        )
        if showPlot or savePlot:
            self.visualize.draw_plot(startNumEpoch=len(self.train_loss_ls) // 5, isShow=showPlot, isSave=savePlot)
            print(str_format("Compelete generate plot !!", fore='g'))
        if saveCSV:
            self.visualize.save_history_csv()
            print(str_format("Compelete generate csv !!", fore='g'))


class DL_Model(DL_Config):
    def __init__(self, device: str = get_device(), **kwargs) -> None:
        try:  # form pre-train model
            self.performance = self.performance
        except AttributeError:  # new model
            super().__init__(**kwargs)
            self.performance = DL_Performance()

        self.device = device

        if self.saveModel:
            self.updateSaveDir = False

        self.epoch_start = len(self.performance.train_loss_ls)
        self.TOTAL_EPOCH = self.epoch_start + self.NUM_EPOCH

        self.train_acc = 0.0
        self.train_loss = 0.0
        self.val_acc = 0.0
        self.val_loss = 0.0

    def training(self, loader: DataLoader, val_loader: DataLoader or None = None, saveModel: bool = False):
        # training
        for self.epoch in range(self.epoch_start, self.TOTAL_EPOCH):
            start_time = time.time()
            num_right = 0
            sum_loss = 0.0

            self.net.train()
            for data, label in loader:
                data = data.to(self.device)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                pred = self.net(data)
                loss = self.loss_func(pred, label)
                loss.backward()
                self.optimizer.step()
                if self.lr_scheduler != None:
                    self.lr_scheduler.step()

                # calculate acc & loss
                sum_loss += loss.item()

                if self.isClassified:
                    preds_result = torch.argmax(pred, dim=1).cpu()

                    num_right += sum(preds_result == label.cpu()).numpy()
                    # num_right += torch.mean((preds_result == label).float()).item()

            self.train_loss = sum_loss / len(loader.dataset)
            self.performance.train_loss_ls.append(self.train_loss)

            if self.isClassified:
                self.train_acc = num_right / len(loader.dataset)  # loader.sampler.num_samples
                self.performance.train_acc_ls.append(self.train_acc)

            if self.printPerformance:
                if val_loader is not None:
                    self.performance.print(self.epoch, self.TOTAL_EPOCH, start_time, end='\r')  # print training info. first~
                    self.validating(val_loader)
                self.performance.print(self.epoch, self.TOTAL_EPOCH, start_time)  # print all info.~
            elif val_loader is not None:
                self.validating(val_loader)

            # save model
            if saveModel:
                self.saveModel = True
                self.updateSaveDir = False
            if self.saveModel:
                self.save_process()

            # early stop
            if self.earlyStop is not None:
                if self.earlyStop == self.epoch - max(self.performance.best_acc_epoch, self.performance.best_loss_epoch):
                    self.TOTAL_EPOCH = self.epoch + 1
                    print(str_format("Early Stop active!!", fore='y', style='blink'))

            # when early stop or outside control happen
            if self.TOTAL_EPOCH == self.epoch + 1:
                self.save_process()
                break

        self.performance.visualize_info(
            showPlot=self.showPlot,
            savePlot=self.savePlot,
            saveCSV=self.savePerformance,
            saveDir=self.saveDir,
        )
        return True

    def validating(self, loader: DataLoader):
        num_right = 0
        sum_loss = 0.0

        self.net.eval()  # change model to the evaluation(val or test) mode.
        with no_grad():
            for data, label in loader:
                data = data.to(self.device)
                label = label.to(self.device)

                # validating process
                pred = self.net(data)
                loss = self.loss_func(pred, label)

                # calculate loss
                sum_loss += loss.item()

                if self.isClassified:
                    pred_label = torch.argmax(pred, dim=1).cpu()
                    num_right += sum(pred_label == label.cpu()).numpy()
                    # num_right += torch.mean((pred_label == label).float()).item()

            # valiation info. record
            self.val_loss = sum_loss / len(loader.dataset)
            self.performance.val_loss_ls.append(self.val_loss)
            if self.performance.val_loss_ls[self.performance.best_loss_epoch] > self.val_loss:
                self.performance.best_loss_epoch = self.epoch

            if self.isClassified:
                self.val_acc = num_right / len(loader.dataset)
                self.performance.val_acc_ls.append(self.val_acc)
                if self.performance.val_acc_ls[self.performance.best_acc_epoch] < self.val_acc:
                    self.performance.best_acc_epoch = self.epoch

    def testing(self, loader: DataLoader):
        self.net.eval()
        result_ls = np.array([])
        with no_grad():
            for data in loader:
                data = data.to(self.device)

                results = self.net(data).cpu()
                if self.isClassified:
                    results = torch.argmax(results, dim=1).numpy()

                result_ls = np.concatenate((result_ls, results), axis=0)

            return result_ls

    def create_saveDir(self):
        # loss_func is from pytorch api or make it by self.
        loss_func_name = (
            self.loss_func.__class__.__name__ if self.loss_func.__class__.__name__ != 'method' else self.loss_func.__name__
        )

        # optimizer is from pytorch api or make it by self.
        optimizer_name = self.optimizer.__class__.__name__ if self.optimizer.__class__.__name__ != 'type' else self.optimizer.__name__

        # generate directory by '{date}-{time}_{model}_{loss-func}_{optimizer}-{lr}_{batch-size}'
        self.saveDir = f'{self.saveDir}{time.strftime("%m%d-%H%M")}_{self.net.__class__.__name__}_{loss_func_name}_{optimizer_name}-{self.optimizer.defaults["lr"]:.1e}_BS-{self.BATCH_SIZE}'

        if self.saveDir is None:
            raise ProcessLookupError(str_format("Need to type path in saveDir from class: DL_Config", fore='r'))

        try:
            if not os.path.exists(self.saveDir):
                os.mkdir(self.saveDir)
                print(str_format(f"Successfully created the directory: {self.saveDir}", fore='g'))
        except OSError:
            raise OSError(str_format(f"Fail to create the directory {self.saveDir} !", fore='r'))

        self.updateSaveDir = True

    def save_process(self):
        if self.updateSaveDir is False:
            self.create_saveDir()
            save_pickle(self.extraHyperConfig, path=f'{self.saveDir}/extraHyperConfig.pickle')
            with open(f'{self.saveDir}/extraHyperConfig.json', "w") as fp:
                json.dump(aa, fp, sort_keys=True, indent=4, default=lambda obj: str(obj))

        # make a parameter mark for model name, if has val_loader in the epoch, use val_loss, else use train_loss
        model_parameter_mark = self.val_loss if self.val_loss != 0.0 else self.train_loss

        # checkpoint
        if self.checkpoint > 0 and (self.epoch + 1) % self.checkpoint == 0:
            self.save_model(f'{self.saveDir}/e{self.epoch+1:03d}_{model_parameter_mark:.3e}.pickle')
        # final epoch
        elif self.epoch + 1 == self.TOTAL_EPOCH:
            self.save_model(f'{self.saveDir}/final_e{self.epoch+1:03d}_{model_parameter_mark:.3e}.pickle')

        # best model
        if self.bestModelSave and self.epoch > 0:
            for key, best_epoch in {'acc': self.performance.best_acc_epoch, 'loss': self.performance.best_loss_epoch}.items():
                if self.epoch == best_epoch:
                    [os.remove(filename) for filename in glob.glob(f'{self.saveDir}/best-{key}*.pickle')]
                    self.save_model(f'{self.saveDir}/best-{key}_e{self.epoch+1:03d}_{model_parameter_mark:.3e}.pickle')

    def save_model(self, path):
        if self.onlyParameters:
            net = self.net
            optimizer = self.optimizer
            lr_scheduler = self.lr_scheduler
            self.net = self.net.state_dict()
            self.optimizer = self.optimizer.state_dict()
            if self.lr_scheduler != None:
                self.lr_scheduler = self.lr_scheduler.state_dict()
            torch.save(self, path)
            self.net = net
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        else:
            torch.save(self, path)

    def load_model(self, path, fullNet=False):
        model: DL_Model = torch.load(path)
        self.performance = model.performance
        self.performance.best_acc_epoch = 0
        self.performance.best_loss_epoch = 0
        self.saveDir = path[: path.rfind('/') + 1]

        self.__init__()

        if fullNet:
            self.net = model.net
            self.optimizer = model.optimizer
            self.lr_scheduler = model.lr_scheduler
        else:
            self.net.load_state_dict(model.net)
            self.optimizer.load_state_dict(model.optimizer)
            if self.lr_scheduler != None:
                self.lr_scheduler.load_state_dict(model.lr_scheduler)
        self.net.eval()


if __name__ == '__main__':
    aa = DL_Model()
    aa.training(None)
