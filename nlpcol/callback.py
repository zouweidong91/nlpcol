
from typing import List

class Callback(object):
    '''Callback基类
    '''
    def __init__(self):
        self.trainer = None  # trainer
        self.model = None  # nn.Module模型，或者包含Trainer的nn.Module
        self.optimizer = None  # 优化器

    def set_params(self, params):
        self.params = params

    def set_trainer(self, trainer):
        self.trainer = trainer

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, global_step, epoch, logs=None):
        pass

    def on_epoch_end(self, global_step, epoch, logs=None):
        pass

    def on_batch_begin(self, global_step, local_step, logs=None):
        pass

    def on_batch_end(self, global_step, local_step, logs=None):
        pass

    def on_dataloader_end(self, logs=None):
        pass

    def on_train_step_end(self, logs=None):
        pass


class CallbackList:
    def __init__(self, callbacks):
        self.callbacks: List[Callback] = callbacks

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, global_step, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(global_step, epoch, logs)

    def on_epoch_end(self, global_step, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(global_step, epoch, logs)

    def on_batch_begin(self, global_step, local_step, logs=None):
        for callback in self.callbacks:
            callback.on_batch_begin(global_step, local_step, logs)

    def on_batch_end(self, global_step, local_step, logs=None):
        for callback in self.callbacks:
            callback.on_batch_end(global_step, local_step, logs)

    def on_dataloader_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_dataloader_end(logs)

    def on_train_step_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_step_end(logs)

        
        