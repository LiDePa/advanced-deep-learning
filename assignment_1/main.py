# -*- coding: utf-8 -*-

from dataset import get_simpsons_subsets


def main(dataset_path, model_name, epochs, weights=None):
    # TODO: setup datasets and call training routine
    get_simpsons_subsets(dataset_path)


if __name__ == '__main__':
    # TODO: start training
    dataset_path = "imgs"
    model_name = ""
    epochs = ""
    weights = None
    main(dataset_path, model_name, epochs, weights)
