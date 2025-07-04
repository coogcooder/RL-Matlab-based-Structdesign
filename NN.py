import torch
import torch.nn as nn
from torch.utils import data as data
from module import parameters as para




class NN1(nn.Module):
    def __init__(self,obs_num,acts_num):
        super(NN1, self).__init__()
        
        self.fc1 = nn.Linear(obs_num, para.HIDDEN_SIZE1)
        self.fc3 = nn.Linear(para.HIDDEN_SIZE1, acts_num)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=1)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1)
        nn.init.normal_(self.fc3.bias, mean=0.0, std=1)
    def __call__(self, x):
        h = torch.tanh(self.fc1(x))
        y = self.fc3(h)        
        return y
    
class NN2(nn.Module):
    def __init__(self,obs_num,acts_num):
        super(NN2, self).__init__()
        
        self.fc1 = nn.Linear(obs_num, para.HIDDEN_SIZE2)
        self.fc3 = nn.Linear(para.HIDDEN_SIZE2, acts_num)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=1)       
        nn.init.normal_(self.fc3.weight, mean=0.0, std=1)
        nn.init.normal_(self.fc3.bias, mean=0.0, std=1)
    def __call__(self, x):
        h = torch.sigmoid(self.fc1(x))
        y = self.fc3(h)
        return y       



class MyDataset(data.Dataset):
    def __init__(self,data1,data2,data3,data4,data5,sample):
        self.pS_dim = data1
        self.S_dim = data2
        self.pS_el = data3
        self.S_el = data4
        self.assets = data5        
        self.sample = sample
    def __getitem__(self, index):
        return self.pS_dim[index] ,self.S_dim[index],self.pS_el[index],self.S_el[index],self.assets[index],self.sample[index]
    def __len__(self):
        return len(self.pS_dim)    