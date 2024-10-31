# -*- coding: utf-8 -*-

import torch

from dataset import get_simpsons_subsets


def main(dataset_path, model_name, epochs, weights=None):
    # TODO: setup datasets and call training routine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_simpsons_subsets(dataset_path)


if __name__ == '__main__':
    # TODO: start training
    dataset_path = "../datasets/simpsons"
    model_name = ""
    epochs = ""
    weights = None
    main(dataset_path, model_name, epochs, weights)
