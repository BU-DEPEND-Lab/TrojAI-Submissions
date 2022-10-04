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

#Input n x K x K x ninput
#Output n x K x 2nh
class same_diff_encoder(nn.Module):
    def __init__(self,ninput,nh,nlayers):
        super().__init__()
        self.encoder=MLP(ninput,nh,nh,nlayers);
    
    def forward(self,fv):
        rounds=fv.shape[0];
        nclasses=fv.shape[1];
        assert fv.shape[2]==nclasses;
        
        h=fv.view(rounds*nclasses*nclasses,-1);
        h=self.encoder(h);
        h=h.view(rounds,nclasses*nclasses,-1);
        
        ind_diag=list(range(0,nclasses*nclasses,nclasses+1));
        ind_off_diag=list(set(list(range(nclasses*nclasses))).difference(set(ind_diag)))
        ind_diag=torch.LongTensor(list(ind_diag)).to(h.device)
        ind_off_diag=torch.LongTensor(list(ind_off_diag)).to(h.device)
        h_diag=h[:,ind_diag,:];
        h_off_diag=h[:,ind_off_diag,:].contiguous().view(rounds,nclasses,nclasses-1,-1).mean(2);
        return torch.cat((h_diag,h_off_diag),dim=2);

#Input n x K x ninput
#Output n x nh
class encoder(nn.Module):
    def __init__(self,ninput,nh,nlayers):
        super().__init__()
        self.ninput=ninput;
        self.encoder=MLP(ninput,nh,nh,nlayers);
    
    def forward(self,fvs):
        maxl=max([x.shape[0] for x in fvs]);
        batch=len(fvs);
        
        data=torch.Tensor(batch,maxl,self.ninput).fill_(0).to(fvs[0].device);
        mask=torch.Tensor(batch,maxl,1).fill_(0).to(fvs[0].device);
        for i,x in enumerate(fvs):
            l=x.shape[0]
            data[i,:l,:]=x;
            mask[i,:l]=1;
        
        h=data.view(batch*maxl,-1);
        h=self.encoder(h);
        h=h.view(batch,maxl,-1);
        
        h=(h*mask).sum(1);
        n=mask.sum(1)+1e-6;
        h=h/n;
        return h;


class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        nh=params.nh;
        nh2=params.nh2;
        nlayers=params.nlayers
        nlayers2=params.nlayers2
        
        try:
            self.margin=params.margin
        except:
            self.margin=8;
        
        bins=100
        self.encoder_hist=encoder(bins*6,nh,nlayers);
        
        self.encoder_combined=MLP(nh,nh2,2,nlayers2);
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        
        return;
    
    def forward(self,data_batch):
        h=[];
        #Have to process one by one due to variable nim & nclasses
        h=self.encoder_hist([x.cuda() for x in data_batch['fvs_eig']])
        h=self.encoder_combined(h);
        h=torch.tanh(h)*self.margin;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    