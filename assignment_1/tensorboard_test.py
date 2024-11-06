import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

#TODO: delete this file!

writer = SummaryWriter("logs")

for n_iter in range(100):
    writer.add_scalar('test1', np.random.random(), n_iter)
    writer.add_scalar('test2', np.random.random(), n_iter)