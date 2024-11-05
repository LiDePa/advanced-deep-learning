# -*- coding: utf-8 -*-

import torch
from dataset import get_dataloader
from training import train
from models import ResNet18Model
#TODO: delete unnecessary imports


def main(dataset_path, model_name, epochs, weights=None):
    # TODO: setup datasets and call training routine
    #TODO: send to device (gpu)
    #dataset_train = SimpsonsDataset(get_simpsons_subsets(dataset_path)[0],get_simpsons_subsets(dataset_path)[1],get_simpsons_subsets(dataset_path)[4],0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on " + device.type)

    train_dataloader = get_dataloader(dataset_path)[0] #this one needs iter?
    one_fc_model = ResNet18Model(37)
    optimizer = torch.optim.SGD(one_fc_model.backbone.parameters())
    train(one_fc_model, optimizer,train_dataloader, device, epochs)


    #train_batch_iter=iter(get_dataloader(dataset_path)[0])


if __name__ == '__main__':
    # TODO: start training
    dataset_path = "../datasets/simpsons"
    model_name = ""
    epochs = 10
    weights = None
    main(dataset_path, model_name, epochs, weights)
