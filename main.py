import torch
from DataLoading import DataHandler
from torchvision import datasets

if __name__ == '__main__' :
    datahandler = DataHandler(datasets=datasets.FashionMNIST) 
    