# -*- coding: utf-8 -*-

import torch

from dataset import get_dataloader
from dataset import SimpsonsDataset

#TODO: delete unnecessary imports


def main(dataset_path, model_name, epochs, weights=None):
    # TODO: setup datasets and call training routine
    #TODO: send to device (gpu)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #dataset_train = SimpsonsDataset(get_simpsons_subsets(dataset_path)[0],get_simpsons_subsets(dataset_path)[1],get_simpsons_subsets(dataset_path)[4],0)

    #train_batch_iter=iter(get_dataloader(dataset_path)[0])

    print()


if __name__ == '__main__':
    # TODO: start training
    dataset_path = "../datasets/simpsons"
    model_name = ""
    epochs = ""
    weights = None
    main(dataset_path, model_name, epochs, weights)
