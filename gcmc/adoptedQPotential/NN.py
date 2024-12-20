import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class EnergyPredictor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(EnergyPredictor, self).__init__()
        self.fc1 = nn.Linear( input_size,    hidden_size1, bias=False)
        self.fc2 = nn.Linear( hidden_size1,  hidden_size2, bias=False)
        self.fc3 = nn.Linear( hidden_size2,  hidden_size2, bias=False)
        self.fc4 = nn.Linear( hidden_size2,  1,            bias=False)
        #self.sig = nn.Sigmoid()
        #self.logsig = nn.LogSigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x =  self.fc1(x)
        x =  self.tanh(x)
        x =  self.fc2(x)
        x =  self.tanh(x)
        x =  self.fc3(x)
        x =  self.tanh(x)
        x =  self.fc4(x)
        #x =  torch.mul(x,2000)
        return x
