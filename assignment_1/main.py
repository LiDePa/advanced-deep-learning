# -*- coding: utf-8 -*-

import torch

from dataset import get_simpsons_subsets
from dataset import SimpsonsDataset


def main(dataset_path, model_name, epochs, weights=None):
    # TODO: setup datasets and call training routine
    #TODO: where do i put the device= stuff?

    dataset = SimpsonsDataset(get_simpsons_subsets(dataset_path)[0],get_simpsons_subsets(dataset_path)[1],get_simpsons_subsets(dataset_path)[4],0)
    print(dataset[1])


if __name__ == '__main__':
    # TODO: start training
    dataset_path = "../datasets/simpsons"
    model_name = ""
    epochs = ""
    weights = None
    main(dataset_path, model_name, epochs, weights)
