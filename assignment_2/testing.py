# TODO: delete this file

from .carla_dataset import get_carla_dataset
from .utils import collate_fn
from .transforms import Normalize
import torch

root = 'datasets/carla3.0_for_students'



a = torch.tensor(1)

mylist = [{"x": a, "y": a}, {"x": a,"y": a}, {"x": a,"y": a}]




nm = Normalize()

sample = get_carla_dataset(root,split="train")[0]["y"]

print(sample)