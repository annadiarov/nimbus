"""
Cosine Warmup Scheduler extracted from the following repository:
    https://github.com/TheMody/Improving-Line-Search-Methods-for-Large-Scale-Neural-Network-Training.git
"""
import numpy as np
import torch.optim as optim


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: optim.Optimizer,
                 warmup: int,
                 n_iterations_cycle: int):
        """
        Cosine learning rate scheduler with warmup. The learning rate is increased
        linearly from 0 to the initial learning rate during the warmup period.
        After the warmup period, the learning rate follows a cosine annealing
        schedule which is restarted every `n_iterations_cycle` epochs.
        :param optimizer: optim.Optimizer
            The optimizer for which to schedule the learning rate.
        :param warmup: int
            Number of warmup iterations to gradually increase the learning rate
            to the initial learning rate.
        :param n_iterations_cycle: int
            The number of epochs between two warm restarts.
        """
        self.warmup = warmup
        self.ti = n_iterations_cycle
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """
        Get the learning rate factor for the given epoch.
        :param epoch: int
            The epoch for which to get the learning rate factor.
        :return:
            The learning rate factor for the given epoch.
        """
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.ti))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
