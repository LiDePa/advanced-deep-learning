import os
import random
from typing import Dict, List
import numpy as np
import torch


def set_deterministic(seed=2408):
    # settings based on https://pytorch.org/docs/stable/notes/randomness.html   Stand 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id):
    # taken from https://pytorch.org/docs/stable/notes/randomness.html
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_model_training_folder(writer):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'weights')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)

def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collates all entries under the same key name into a entry with the same name in a new dictionary.

    Args:
        samples (List[Dict[str, torch.Tensor]]): A list of data samples.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of stacked data samples.
    """

    # TODO: Stack the entries of every data sample into a single tensor.
    #       Keep the same keys.

    raise NotImplementedError("utils.collate_fn has nbt been implemented yet.")