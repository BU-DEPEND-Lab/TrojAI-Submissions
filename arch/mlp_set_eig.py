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
        self.bn=nn.LayerNorm(ninput);
        if nlayers==1:
            self.layers.append(nn.Linear(ninput,noutput));
        else:
            self.layers.append(nn.Linear(ninput,nh));
            for i in range(nlayers-2):
                self.layers.append(nn.Linear(nh,nh));
            
            self.layers.append(nn.Linear(nh,noutput));
        
        self.ninput=ninput;
        self.noutput=noutput;
        return;
    
    def forward(self,x):
        h=x.view(-1,self.ninput);
        #h=self.bn(h);
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.relu(h);
            #h=F.dropout(h,training=self.training);
        
        h=self.layers[-1](h);
        return h


class new(nn.Module):
    def __init__(self,params,input_size=None):
        super(new,self).__init__()
        nh=10 * params.nh;
        nh3=100 * params.nh3;
        nlayers=params.nlayers
        nlayers2=params.nlayers2
        
        q=int((params.nh2//2)**0.5);
        self.q=torch.arange(0,1+1e-8,1/q).cuda()
        q=len(self.q);
        
        try:
            self.margin=params.margin
        except:
            self.margin=8;
        
        bins=100
        in_shape = 0
        if input_size is not None:
            in_shape = input_size
        else:
            in_shape = bins*6 
        in_shape = 0
        self.encoder_hist=MLP(in_shape + 20,nh,nh,nlayers);
        self.encoder_combined=MLP(100*nh,nh3,2,nlayers2);
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        return;
    
    def forward(self,data_batch):
        weight_dist=data_batch['fvs'];
        b=len(weight_dist);
        h=[];
        #Have to process one by one due to variable nim & nclasses
        for i in range(b):
            h_i=self.encoder_hist(weight_dist[i].cuda());
            #print('before quantile', h_i, h_i.shape)
            #h_i=torch.quantile(h_i,self.q,dim=0).contiguous().view(-1);
            #print('quantile', h_i, h_i.shape)
            h_i = h_i.flatten()
            h_i=self.encoder_combined(h_i.unsqueeze(0)).squeeze(0);
            h.append(h_i);
        
        h=torch.stack(h,dim=0);
        #print('before combined MLP', h.shape)
        #h=self.encoder_combined(h);
        #print('after combined MLP', h.shape)
        h=torch.tanh(h)*self.margin;
        #print('output after passing to tanh', h.shape)
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    
