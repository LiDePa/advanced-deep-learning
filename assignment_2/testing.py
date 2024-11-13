# TODO: delete this file

from .carla_dataset import get_carla_dataset
from .utils import collate_fn
import torch

a = torch.tensor(1)

mylist = [{"x": a, "y": a}, {"x": a,"y": a}, {"x": a,"y": a}]

root = 'datasets/carla3.0_for_students'
print(collate_fn(mylist))