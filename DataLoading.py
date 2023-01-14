import torch 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets






t_dataset = datasets.FashionMNIST(root="./data",train=True, download=False,transform=ToTensor())
v_dataset = datasets.FashionMNIST(root="./data",train=False,download=False,transform=ToTensor())
batch_size = 64 
t_data = DataLoader(t_dataset,batch_size=batch_size,shuffle=True)
v_data = DataLoader(v_dataset,batch_size=batch_size,shuffle=True)