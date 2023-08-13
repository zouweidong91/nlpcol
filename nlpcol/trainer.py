
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nlpcol.callback import Callback, CallbackList


class TrainConfig:
    """训练超参数配置
    """
    def __init__(self, *args, **kwargs):
        self.batch_size:int = kwargs.get("batch_size", 8)
        self.epochs:int = kwargs.get("epochs", 3)
        self.max_seq_length:int = kwargs.get("max_seq_length", 128)


class Trainer:

    def __init__(
        self,
        model: nn.Module, 
        config: TrainConfig,
        loss_fn: Callable,
        optimizer: Optimizer,
        collate_fn: Callable,
        lr_scheduler: LambdaLR = None
    ):
        """简单实现的通用trainer类

        Args:
            model (nn.Module): 模型类
            loss_fn (Callable): 损失函数
            optimizer (torch.optim.Optimizer): 优化器.
            collate_fn (Callable): data_loader时数据校验函数.
            lr_scheduler (LambdaLR, optional): 学习率调整器. Defaults to None.
        """
        
        self.model:nn.Module = model
        self.config = config
        self.loss_fn = loss_fn
        self.optimizer:Optimizer = optimizer
        self.collate_fn = collate_fn
        self.lr_scheduler = lr_scheduler

        self.global_step = 0 # 当前全局step
        self.local_step = 0  # epoch内当前step
        self.total_steps = 0 # 所有的steps
        self.epoch = 0       # 当前epoch

    def update_global_step(self):
        """更新global_step
        """
        self.global_step = self.epoch * self.steps_per_epoch + self.local_step

    def get_dataloader(self, dataset, shuffle:bool=False):
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, \
            shuffle=shuffle, collate_fn=self.collate_fn)
        return dataloader
    
    def args_expend(self, X):
        """model入参是否展开"""
        if isinstance(X, torch.Tensor):
            return False
        elif type(X) in [list, tuple]:
            return True
        return False


    def train_step(self, batch):
        """每个batch的执行逻辑 返回损失

        Args:
            batch (_type_): batch数据  X: list  y:tensor
        """
        self.update_global_step()
        self.callbacks.on_batch_begin(self.global_step, self.local_step)

        X, y = batch
        output = self.model(*X) if self.args_expend(X) else self.model(X)
        loss:torch.Tensor = self.loss_fn(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.callbacks.on_batch_end(self.global_step, self.local_step)
        return loss
    
    def train_loop(self, train_dataloader):
        """每个epoch的执行逻辑

        Args:
            epoch (int): 当前epoch
        """
        self.local_step = 0
        self.update_global_step()
        self.callbacks.on_epoch_begin(self.global_step, self.epoch)
        progress_bar = tqdm(range(len(train_dataloader)))
        progress_bar.set_description(f'loss: {0:>7f}')
        # model.train()的作用是启用 Layer Normalization 和 Dropout。
        # 保证LN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
        self.model.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            self.local_step = step
            loss = self.train_step(batch)
            epoch_loss += loss.item()
            progress_bar.set_description(f'loss: {loss:>7f}')
            progress_bar.update(1)

        avg_train_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {self.epoch+1} avg_train_loss: {avg_train_loss}")
        self.callbacks.on_epoch_end(self.global_step, self.epoch+1)


    def train(self, train_dataset, callbacks: List[Callback] = None):
        """执行训练主函数
            train_dataset (Dataset): 训练数据.
            callbacks (List[Callback], optional): 回调类. Defaults to None.
        """
        self.callbacks = CallbackList(callbacks or [])
        epochs = self.config.epochs
        train_dataloader = self.get_dataloader(train_dataset, shuffle=True)

        self.steps_per_epoch = len(train_dataloader)
        self.total_steps = self.steps_per_epoch * epochs

        self.callbacks.on_train_begin()

        for epoch in range(epochs):
            self.epoch = epoch
            print(f"Epoch {epoch+1}/{epochs}\n-------------------------------")
            self.train_loop(train_dataloader)

        self.callbacks.on_train_end()
        print("Train Done")
        
    @torch.no_grad()
    def predict(self, X:list):
        # model.eval() 不启用 Batch Normalization 和 Dropout。
        self.model.eval()
        output = self.model(*X)
        return output


    def load_weights(self, load_path, strict=True):
        """加载模型权重

        Args:
            load_path (_type_): 权重路径
        """
        state_dict = torch.load(load_path)
        self.model.load_state_dict(state_dict, strict)


    def save_weights(self, save_path):
        """保存模型权重

        Args:
            save_path (_type_): 保存路径
        """
        state_dict = self.model.state_dict()
        torch.save(state_dict, save_path)
    