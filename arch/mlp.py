import torch
import torchvision.models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy

class MLP(nn.Module):
    def __init__(self,ninput,nh,noutput,nlayers):
        super().__init__()
        print('inside MLP ninput' + str(ninput))
        print('inside MLP nouput' + str(noutput))

        self.layers=nn.ModuleList();
        if nlayers==1:
            self.layers.append(nn.Linear(ninput,noutput));
        else:
            self.layers.append(nn.Linear(ninput,nh));
            for i in range(nlayers-2):
                self.layers.append(nn.Linear(nh,nh));
            
            self.layers.append(nn.Linear(nh,noutput));
        
        self.ninput=ninput;
        self.noutput=noutput;
        self.sigmoid = nn.Sigmoid()
        return;
    
    def forward(self,x):
        h=x.view(-1,self.ninput);
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.relu(h);
        
        h=self.layers[-1](h);
        return h


class new(nn.Module):
    def __init__(self, params, input_size=None, output_size=1):
        super(new,self).__init__()
        nh=params.nh;
        nh3=params.nh3;
        nlayers=params.nlayers
        nlayers2=params.nlayers2
        
        bins=100
        in_shape = 0
        if input_size is not None:
            in_shape = input_size
        else:
            in_shape = bins*6 
        self.mlp=MLP(100 * 20,nh,output_size,nlayers);
        return;
    
    def forward(self,data_batch):
        x = data_batch['fvs'];
        logits = self.mlp(torch.stack(x).cuda());
        
        return logits.view(-1)
    def logp(self, data_batch):

        return nn.Sigmoid(self.forward(data_batch))