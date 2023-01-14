import torch 
from torch import nn 

class LinearRegression(nn.Module) :
    
    def __init__(self, input_size, hidden_size, output_size) :
        super(LinearRegression,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_1 = nn.Linear(self.input_size,  self.hidden_size)
        self.layer_2 = nn.Linear(self.hidden_size, self.output_size)
        self.softmax_head  = nn.Sequential(nn.Flatten() , self.layer_1, self.layer_2, nn.Softmax(dim=1))
    
    def forward(self, x) :
        x = self.softmax_head(x)
        return x
        