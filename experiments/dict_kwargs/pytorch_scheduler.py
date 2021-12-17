# Goal: self.scheduler = scheduler(self.optimizer, **scheduler_config)

# api: torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=- 1, verbose=False)
# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR 

import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.parameter import Parameter
from torch.optim import SGD

model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)

# python pass dict as kwargs
# https://stackoverflow.com/a/5710402/7748163
scheduler_conf = {'gamma': '0.1'}

lr_sched = ExponentialLR(optimizer, **scheduler_conf)
