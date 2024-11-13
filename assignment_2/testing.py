# TODO: delete this file

from .carla_dataset import get_carla_dataset

root = 'datasets/carla3.0_for_students'

print(get_carla_dataset(root, split="train", transforms=[]).label_paths)