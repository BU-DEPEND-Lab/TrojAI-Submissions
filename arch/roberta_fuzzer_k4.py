import torch
import torch.linalg
import torchvision.models
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy
import time

import torch.optim as optim

from transformers import RobertaTokenizer, RobertaForMaskedLM

class Fuzzer():
    def __init__(self):
        n=0;
        for n in range(5):
            try:
                self.tokenizer=torch.load('roberta_tokenizer.pt');
                break
            except:
                try:
                    self.tokenizer=torch.load('/roberta_tokenizer.pt');
                    break
                except:
                    try:
                        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base");
                        torch.save(self.tokenizer,'roberta_tokenizer.pt')
                        break
                    except:
                        print('Failed to load tokenizer for fuzzer, try again');
                        pass;
        
        self.pad=self.encode(self.tokenizer.pad_token)[0];
        self.vocab_size=len(self.tokenizer);
        self.special_tokens=set([self.encode(x)[0] for x in [self.tokenizer.bos_token,self.tokenizer.eos_token,self.tokenizer.unk_token,self.tokenizer.sep_token,self.tokenizer.pad_token,self.tokenizer.cls_token,self.tokenizer.mask_token,]+self.tokenizer.additional_special_tokens]);
        self.valid_tokens=list(set(range(self.vocab_size)).difference(self.special_tokens));
        
        if not surrogate is None:
            self.surrogate=surrogate;
        
    
    def encode(self,data):
        if isinstance(data,list):
            return [self.tokenizer.encode(s)[1:-1] for s in data];
        elif isinstance(data,str):
            return self.tokenizer.encode(data)[1:-1];
        else:
            print('unrecognized input');
            a=0/0;
    
    def decode(self,data):
        if isinstance(data,list):
            if len(data)==0:
                return self.tokenizer.decode(data);
            elif isinstance(data[0],list):
                return [self.tokenizer.decode([t for t in s if not t==self.pad]) for s in data];
            else:
                return self.tokenizer.decode([t for t in data if not t==self.pad]);
        elif torch.is_tensor(data):
            return self.decode(data.tolist());
        else:
            print('unrecognized input');
            a=0/0;
    
    def generate_random_sequence(self,length=1,decode=False,n=-1,maxl=None):
        valid_tokens=torch.LongTensor(self.valid_tokens);
        if maxl is None:
            maxl=length;
        
        if n>=1:
            x=torch.LongTensor(n,length).random_(len(self.valid_tokens));
        else:
            x=torch.LongTensor(length).random_(len(self.valid_tokens));
        
        tokens=valid_tokens[x];
        tokens=F.pad(tokens,(0,maxl-length),value=self.pad);
        
        if decode:
            tokens=self.decode(tokens)
        return tokens;
    
    def create_surrogate(self,maxl=8):
        #Load initial word embeddings
        n=0;
        for n in range(5):
            try:
                roberta=torch.load('roberta_model.pt');
                break
            except:
                try:
                    roberta=torch.load('/roberta_model.pt');
                    break
                except:
                    try:
                        roberta=RobertaForMaskedLM.from_pretrained('roberta-base');
                        torch.save(roberta,'roberta_model.pt')
                        break
                    except:
                        print('Failed to load embeddings for fuzzer, try again');
                        pass;
        
        
        
        we=roberta.roberta.embeddings.word_embeddings.weight;
        return surrogate(we,maxl=maxl);
    
    def find_min(self,x,y,l=8,w=None):
        #Assuming surrogate has registered we
        w=w.type(self.surrogate.reg.dtype)
        y=y.type(self.surrogate.reg.dtype)
        
        nobs=y.shape[-1];
        if w is None:
            w=torch.Tensor(nobs).fill_(1/nobs).cuda();
        else:
            w=w.cuda().view(-1,1);
        
        #Calibrate uncertainty from the surrogate
        #Using 50% for training
        n=len(x);
        ind=torch.randperm(n);
        ind_train=ind[:n//2];
        ind_test=ind[n//2:]
        
        ws=self.surrogate.regress(x[ind_train],y[ind_train]);
        pred_y,pred_y_std=self.surrogate.score(x[ind_test],*ws);
        z=(y[ind_test]-pred_y)/(pred_y_std+1e-8);
        s_cal=torch.log(z.std(dim=0,keepdim=True)+1e-8);
        s_cal=s_cal*n/(300+n); #Prior weight 300
        s_cal=torch.exp(s_cal);
        
        
        ws=self.surrogate.regress(x,y);
        best_scores=[];
        current_best=torch.mm(y,w).squeeze(dim=-1).min();
        
        #Tool for checking what options are available for tokens[ind]
        #Tokens: 1D tensor
        #ind: int
        #dupes: 2D tensor
        #valid_tokens: list
        def check_dupe(tokens,ind,dupes,valid_tokens):
            l=len(tokens);
            q=tokens.clone().to(dupes.device);
            q[ind]=-1;
            #Identify available indicies
            available_tokens=copy.deepcopy(valid_tokens);
            if len(dupes)>0:
                match=(dupes-q.view(1,-1)).eq(0).long().sum(dim=1).ge(l-1).nonzero();
                match=match.view(-1).tolist()
                if len(match)>0:
                    dupe_tokens=[int(dupes[j][ind]) for j in match];
                    available_tokens=list(set(valid_tokens).difference(set(dupe_tokens)));
            
            return available_tokens;
        
        
        with torch.no_grad():
            tokens=torch.LongTensor(self.surrogate.maxl).fill_(self.pad);
            candidates=[];
            for i in range(1,l+1):
                best_score=1e10;
                tokens[0:i]=torch.LongTensor(self.generate_random_sequence(i));
                while True:
                    improved=False
                    for ind in range(0,i):
                        #Generate input with options
                        available_tokens=check_dupe(tokens,ind,x,self.valid_tokens);
                        tokens_=tokens.tolist();
                        tokens_[ind]=torch.LongTensor(available_tokens).cuda();
                        
                        #Evaluate all options
                        qy,qy_std=self.surrogate.score(tokens_,*ws);
                        #qy_std=qy_std*s_cal;
                        
                        qy_avg=torch.mm(qy,w).squeeze(dim=-1)
                        qy_std_avg=torch.mm(qy_std**2,w**2).squeeze(dim=-1)
                        qy_std_avg=qy_std_avg**0.5;
                        
                        #Compute expected improvement
                        #This is minimization version
                        from scipy.stats import norm
                        imp=-qy_avg+current_best
                        z=imp/(qy_std_avg+1e-8);
                        z=z.double()
                        
                        cdfz=0.5*(1+torch.erf(z/math.sqrt(2)));
                        pdfz=1/math.sqrt(2*math.pi)*torch.exp(-0.5*z**2);
                        
                        ei=imp.double()*cdfz+qy_std_avg.double()*pdfz
                        
                        #Find largest expected improvement
                        s=-ei;
                        score,j=s.min(dim=0);
                        score=float(score);
                        
                        if score>=0:
                            print('switching criteria',end='\r')
                            s=-z;
                            score,j=s.min(dim=0);
                            score=float(score);
                        
                        j=int(tokens_[ind][j]);
                        if score<best_score:
                            best_score=score
                            tokens[ind]=j;
                            improved=True;
                    
                    if not improved:
                        break;
                
                candidates.append(tokens.tolist());
                best_scores.append(float(score));
        
        return candidates,best_scores
    
    def find_min_v2(self,x,y,l=8,w=None,l_=2):
        #Assuming surrogate has registered we
        w=w.type(self.surrogate.reg.dtype)
        y=y.type(self.surrogate.reg.dtype)
        
        nobs=y.shape[-1];
        if w is None:
            w=torch.Tensor(nobs).fill_(1/nobs).cuda();
        else:
            w=w.cuda().view(-1,1);
        
        #Calibrate uncertainty from the surrogate
        #Using 50% for training
        n=len(x);
        ind=torch.randperm(n);
        ind_train=ind[:n//2];
        ind_test=ind[n//2:]
        
        #ws=self.surrogate.regress(x[ind_train],y[ind_train]);
        #pred_y,pred_y_std=self.surrogate.score(x[ind_test],*ws);
        #z=(y[ind_test]-pred_y)/(pred_y_std+1e-8);
        #s_cal=torch.log(z.std(dim=0,keepdim=True)+1e-8);
        #s_cal=s_cal*n/(300+n); #Prior weight 300
        #s_cal=torch.exp(s_cal);
        
        
        ws=self.surrogate.regress(x,y);
        best_scores=[];
        current_best,ind_best=torch.mm(y,w).squeeze(dim=-1).min(dim=0);
        current_best=float(current_best);
        ind_best=int(ind_best);
        
        #Tool for checking what options are available for tokens[ind]
        #Tokens: 1D tensor
        #ind: int
        #dupes: 2D tensor
        #valid_tokens: list
        def check_dupe(tokens,ind,dupes,valid_tokens):
            l=len(tokens);
            q=tokens.clone().to(dupes.device);
            q[ind]=-1;
            #Identify available indicies
            available_tokens=copy.deepcopy(valid_tokens);
            if len(dupes)>0:
                match=(dupes-q.view(1,-1)).eq(0).long().sum(dim=1).ge(l-1).nonzero();
                match=match.view(-1).tolist()
                if len(match)>0:
                    dupe_tokens=[int(dupes[j][ind]) for j in match];
                    available_tokens=list(set(valid_tokens).difference(set(dupe_tokens)));
            
            return available_tokens;
        
        
        with torch.no_grad():
            candidates=[];
            for _ in range(6):
                best_score=1e10;
                
                tokens=x[ind_best].clone();
                lbest=int((~tokens.eq(self.pad)).long().sum())
                working_set=min(lbest+1,l);
                p=torch.distributions.Categorical(torch.Tensor([0,0.25,0,0,0,0])[:working_set]);
                i=int(p.sample())+1;
                #i=min(working_set,l_);
                s=torch.LongTensor(self.generate_random_sequence(i)).cuda();
                inds=torch.randperm(working_set)[:i]
                
                tokens[inds]=s;
                
                import sys
                print(lbest,i);
                sys.stdout.flush()
                
                while True:
                    improved=False
                    for ind in inds:
                        #Generate input with options
                        available_tokens=check_dupe(tokens,ind,x,self.valid_tokens);
                        tokens_=tokens.tolist();
                        tokens_[ind]=torch.LongTensor(available_tokens).cuda();
                        
                        #Evaluate all options
                        qy,qy_std=self.surrogate.score(tokens_,*ws);
                        #qy_std=qy_std*s_cal;
                        
                        qy_avg=torch.mm(qy,w).squeeze(dim=-1)
                        qy_std_avg=torch.mm(qy_std**2,w**2).squeeze(dim=-1)
                        qy_std_avg=qy_std_avg**0.5;
                        
                        #Compute expected improvement
                        #This is minimization version
                        from scipy.stats import norm
                        imp=-qy_avg+current_best
                        z=imp/(qy_std_avg+1e-8);
                        z=z.double()
                        
                        cdfz=0.5*(1+torch.erf(z/math.sqrt(2)));
                        pdfz=1/math.sqrt(2*math.pi)*torch.exp(-0.5*z**2);
                        
                        ei=imp.double()*cdfz+qy_std_avg.double()*pdfz
                        
                        #Find largest expected improvement
                        s=-ei;
                        score,j=s.min(dim=0);
                        score=float(score);
                        
                        if score>=0:
                            print('switching criteria',end='\r')
                            s=-z;
                            score,j=s.min(dim=0);
                            score=float(score);
                        
                        j=int(tokens_[ind][j]);
                        if score<best_score:
                            best_score=score
                            tokens[ind]=j;
                            improved=True;
                    
                    if not improved:
                        break;
                
                candidates.append(tokens.tolist());
                best_scores.append(float(score));
        
        return candidates,best_scores
    
    def find_min_v3(self,x,y,l=8,w=None,l_=3):
        #Assuming surrogate has registered we
        w=w.type(self.surrogate.reg.dtype)
        y=y.type(self.surrogate.reg.dtype)
        
        nobs=y.shape[-1];
        if w is None:
            w=torch.Tensor(nobs).fill_(1/nobs).cuda();
        else:
            w=w.cuda().view(-1,1);
        
        #Calibrate uncertainty from the surrogate
        #Using 50% for training
        n=len(x);
        ind=torch.randperm(n);
        ind_train=ind[:n//2];
        ind_test=ind[n//2:]
        
        #ws=self.surrogate.regress(x[ind_train],y[ind_train]);
        #pred_y,pred_y_std=self.surrogate.score(x[ind_test],*ws);
        #z=(y[ind_test]-pred_y)/(pred_y_std+1e-8);
        #s_cal=torch.log(z.std(dim=0,keepdim=True)+1e-8);
        #s_cal=s_cal*n/(300+n); #Prior weight 300
        #s_cal=torch.exp(s_cal);
        
        
        ws=self.surrogate.regress(x,y);
        best_scores=[];
        current_best,ind_best=torch.mm(y,w).squeeze(dim=-1).min(dim=0);
        current_best=float(current_best);
        ind_best=int(ind_best);
        
        #Tool for checking what options are available for tokens[ind]
        #Tokens: 1D tensor
        #ind: int
        #dupes: 2D tensor
        #valid_tokens: list
        def check_dupe(tokens,ind,dupes,valid_tokens):
            l=len(tokens);
            q=tokens.clone().to(dupes.device);
            q[ind]=-1;
            #Identify available indicies
            available_tokens=copy.deepcopy(valid_tokens);
            if len(dupes)>0:
                match=(dupes-q.view(1,-1)).eq(0).long().sum(dim=1).ge(l-1).nonzero();
                match=match.view(-1).tolist()
                if len(match)>0:
                    dupe_tokens=[int(dupes[j][ind]) for j in match];
                    available_tokens=list(set(valid_tokens).difference(set(dupe_tokens)));
            
            return available_tokens;
        
        
        with torch.no_grad():
            candidates=[];
            lbest=int((~x[ind_best].eq(self.pad)).long().sum())
            working_set=min(lbest+1,l);
            for i in range(working_set):
                best_score=1e10;
                
                tokens=x[ind_best].clone();
                s=torch.LongTensor(self.generate_random_sequence(1)).cuda();
                tokens[i]=s;
                
                while True:
                    improved=False
                    for ind in [i]:
                        #Generate input with options
                        available_tokens=check_dupe(tokens,ind,x,self.valid_tokens);
                        tokens_=tokens.tolist();
                        tokens_[ind]=torch.LongTensor(available_tokens).cuda();
                        
                        #Evaluate all options
                        qy,qy_std=self.surrogate.score(tokens_,*ws);
                        #qy_std=qy_std*s_cal;
                        
                        qy_avg=torch.mm(qy,w).squeeze(dim=-1)
                        qy_std_avg=torch.mm(qy_std**2,w**2).squeeze(dim=-1)
                        qy_std_avg=qy_std_avg**0.5;
                        
                        #Compute expected improvement
                        #This is minimization version
                        from scipy.stats import norm
                        imp=-qy_avg+current_best
                        z=imp/(qy_std_avg+1e-8);
                        z=z.double()
                        
                        cdfz=0.5*(1+torch.erf(z/math.sqrt(2)));
                        pdfz=1/math.sqrt(2*math.pi)*torch.exp(-0.5*z**2);
                        
                        ei=imp.double()*cdfz+qy_std_avg.double()*pdfz
                        
                        #Find largest expected improvement
                        s=-ei;
                        score,j=s.min(dim=0);
                        score=float(score);
                        
                        if score>=0:
                            print('switching criteria',end='\r')
                            s=-z;
                            score,j=s.min(dim=0);
                            score=float(score);
                        
                        j=int(tokens_[ind][j]);
                        if score<best_score:
                            best_score=score
                            tokens[ind]=j;
                            improved=True;
                    
                    if not improved:
                        break;
                
                candidates.append(tokens.tolist());
                best_scores.append(float(score));
        
        return candidates,best_scores
    
    def find_min_div(self,x,y,l=8,w=None):
        #Assuming surrogate has registered we
        w=w.type(self.surrogate.reg.dtype)
        y=y.type(self.surrogate.reg.dtype)
        
        nobs=y.shape[-1];
        if w is None:
            w=torch.Tensor(nobs).fill_(1/nobs).cuda();
        else:
            w=w.cuda().view(-1,1);
        
        
        ws=self.surrogate.regress(x,y);
        best_scores=[];
        current_best=torch.mm(y,w).squeeze(dim=-1).min();
        
        #Tool for checking what options are available for tokens[ind]
        #Tokens: 1D tensor
        #ind: int
        #dupes: 2D tensor
        #valid_tokens: list
        def check_dupe(tokens,ind,dupes,valid_tokens):
            l=len(tokens);
            q=tokens.clone().to(dupes.device);
            q[ind]=-1;
            #Identify available indicies
            available_tokens=copy.deepcopy(valid_tokens);
            if len(dupes)>0:
                match=(dupes-q.view(1,-1)).eq(0).long().sum(dim=1).ge(l-1).nonzero();
                match=match.view(-1).tolist()
                if len(match)>0:
                    dupe_tokens=[int(dupes[j][ind]) for j in match];
                    available_tokens=list(set(valid_tokens).difference(set(dupe_tokens)));
            
            return available_tokens;
        
        
        with torch.no_grad():
            tokens=torch.LongTensor(self.surrogate.maxl).fill_(self.pad);
            candidates=[];
            for i in range(1,l+1):
                best_score=1e10;
                tokens[0:i]=torch.LongTensor(self.generate_random_sequence(i));
                while True:
                    improved=False
                    for ind in range(0,i):
                        #Generate input with options
                        available_tokens=check_dupe(tokens,ind,x,self.valid_tokens);
                        tokens_=tokens.tolist();
                        tokens_[ind]=torch.LongTensor(available_tokens).cuda();
                        
                        #Evaluate all options
                        qy,qy_std=self.surrogate.score(tokens_,*ws);
                        #qy_std=qy_std*s_cal;
                        
                        qy_avg=torch.mm(qy,w).squeeze(dim=-1)
                        qy_std_avg=torch.mm(qy_std**2,w**2).squeeze(dim=-1)
                        qy_std_avg=qy_std_avg**0.5;
                        
                        #Find largest expected improvement
                        s=-qy_std_avg;
                        score,j=s.min(dim=0);
                        score=float(score);
                        
                        if score>=0:
                            print('switching criteria',end='\r')
                            s=-z;
                            score,j=s.min(dim=0);
                            score=float(score);
                        
                        j=int(tokens_[ind][j]);
                        if score<best_score:
                            best_score=score
                            tokens[ind]=j;
                            improved=True;
                    
                    if not improved:
                        break;
                
                candidates.append(tokens.tolist());
                best_scores.append(float(score));
        
        return candidates,best_scores
    
    def find_min_div_v2(self,x,y,l=8,w=None,l_=2):
        #Assuming surrogate has registered we
        w=w.type(self.surrogate.reg.dtype)
        y=y.type(self.surrogate.reg.dtype)
        
        nobs=y.shape[-1];
        if w is None:
            w=torch.Tensor(nobs).fill_(1/nobs).cuda();
        else:
            w=w.cuda().view(-1,1);
        
        #Calibrate uncertainty from the surrogate
        #Using 50% for training
        n=len(x);
        ind=torch.randperm(n);
        ind_train=ind[:n//2];
        ind_test=ind[n//2:]
        
        #ws=self.surrogate.regress(x[ind_train],y[ind_train]);
        #pred_y,pred_y_std=self.surrogate.score(x[ind_test],*ws);
        #z=(y[ind_test]-pred_y)/(pred_y_std+1e-8);
        #s_cal=torch.log(z.std(dim=0,keepdim=True)+1e-8);
        #s_cal=s_cal*n/(300+n); #Prior weight 300
        #s_cal=torch.exp(s_cal);
        
        
        ws=self.surrogate.regress(x,y);
        best_scores=[];
        current_best,ind_best=torch.mm(y,w).squeeze(dim=-1).min(dim=0);
        current_best=float(current_best);
        ind_best=int(ind_best);
        
        #Tool for checking what options are available for tokens[ind]
        #Tokens: 1D tensor
        #ind: int
        #dupes: 2D tensor
        #valid_tokens: list
        def check_dupe(tokens,ind,dupes,valid_tokens):
            l=len(tokens);
            q=tokens.clone().to(dupes.device);
            q[ind]=-1;
            #Identify available indicies
            available_tokens=copy.deepcopy(valid_tokens);
            if len(dupes)>0:
                match=(dupes-q.view(1,-1)).eq(0).long().sum(dim=1).ge(l-1).nonzero();
                match=match.view(-1).tolist()
                if len(match)>0:
                    dupe_tokens=[int(dupes[j][ind]) for j in match];
                    available_tokens=list(set(valid_tokens).difference(set(dupe_tokens)));
            
            return available_tokens;
        
        
        with torch.no_grad():
            candidates=[];
            for _ in range(6):
                best_score=1e10;
                
                tokens=x[-1].clone();
                lbest=int((~tokens.eq(self.pad)).long().sum())
                working_set=min(lbest+1,l);
                p=torch.distributions.Categorical(torch.Tensor([0,0.25,0,0,0,0])[:working_set]);
                i=int(p.sample())+1;
                #i=min(working_set,l_);
                s=torch.LongTensor(self.generate_random_sequence(i)).cuda();
                inds=torch.randperm(working_set)[:i]
                
                tokens[inds]=s;
                
                import sys
                print(lbest,i);
                sys.stdout.flush()
                
                while True:
                    improved=False
                    for ind in inds:
                        #Generate input with options
                        available_tokens=check_dupe(tokens,ind,x,self.valid_tokens);
                        tokens_=tokens.tolist();
                        tokens_[ind]=torch.LongTensor(available_tokens).cuda();
                        
                        #Evaluate all options
                        qy,qy_std=self.surrogate.score(tokens_,*ws);
                        #qy_std=qy_std*s_cal;
                        
                        qy_avg=torch.mm(qy,w).squeeze(dim=-1)
                        qy_std_avg=torch.mm(qy_std**2,w**2).squeeze(dim=-1)
                        qy_std_avg=qy_std_avg**0.5;
                        
                        #Find largest expected improvement
                        s=-qy_std_avg;
                        score,j=s.min(dim=0);
                        score=float(score);
                        
                        if score>=0:
                            print('switching criteria',end='\r')
                            s=-z;
                            score,j=s.min(dim=0);
                            score=float(score);
                        
                        j=int(tokens_[ind][j]);
                        if score<best_score:
                            best_score=score
                            tokens[ind]=j;
                            improved=True;
                    
                    if not improved:
                        break;
                
                candidates.append(tokens.tolist());
                best_scores.append(float(score));
        
        return candidates,best_scores
    
    def find_min_baseline(self,x,y,l=8,w=None):
        #Assuming surrogate has registered we
        w=w.type(self.surrogate.reg.dtype)
        y=y.type(self.surrogate.reg.dtype)
        
        nobs=y.shape[-1];
        if w is None:
            w=torch.Tensor(nobs).fill_(1/nobs).cuda();
        else:
            w=w.cuda().view(-1,1);
        
        #Calibrate uncertainty from the surrogate
        #Using 50% for training
        n=len(x);
        ind=torch.randperm(n);
        ind_train=ind[:n//2];
        ind_test=ind[n//2:]
        
        ws=self.surrogate.regress(x[ind_train],y[ind_train]);
        pred_y,pred_y_std=self.surrogate.score(x[ind_test],*ws);
        z=(y[ind_test]-pred_y)/(pred_y_std+1e-8);
        s_cal=torch.log(z.std(dim=0,keepdim=True)+1e-8);
        s_cal=s_cal*n/(300+n); #Prior weight 300
        s_cal=torch.exp(s_cal);
        
        
        ws=self.surrogate.regress(x,y);
        current_best,ind_best=torch.mm(y,w).squeeze(dim=-1).min(dim=0);
        current_best=float(current_best);
        ind_best=int(ind_best);
        
        #Tool for checking what options are available for tokens[ind]
        #Tokens: 1D tensor
        #ind: int
        #dupes: 2D tensor
        #valid_tokens: list
        def check_dupe(tokens,ind,dupes,valid_tokens):
            l=len(tokens);
            q=tokens.clone().to(dupes.device);
            q[ind]=-1;
            #Identify available indicies
            available_tokens=copy.deepcopy(valid_tokens);
            if len(dupes)>0:
                match=(dupes-q.view(1,-1)).eq(0).long().sum(dim=1).ge(l-1).nonzero();
                match=match.view(-1).tolist()
                if len(match)>0:
                    dupe_tokens=[int(dupes[j][ind]) for j in match];
                    available_tokens=list(set(valid_tokens).difference(set(dupe_tokens)));
            
            return available_tokens;
        
        
        with torch.no_grad():
            best_scores=[];
            candidates=[];
            for _ in range(1):
                best_score=1e10;
                
                tokens=x[ind_best].clone();
                lbest=int((~tokens.eq(self.pad)).long().sum())
                working_set=min(lbest+1,l);
                i=int(torch.LongTensor(1).random_(min(min(lbest+1,l),3)))+1;
                s=torch.LongTensor(self.generate_random_sequence(i)).cuda();
                inds=torch.randperm(working_set)[:i]
                
                tokens[inds]=s;
                
                import sys
                print(lbest,i);
                sys.stdout.flush()
                
                candidates.append(tokens.tolist());
                best_scores.append(float(0));
        
        return candidates,best_scores
    
    
    def suggest(self,x,y,l=8,target=None):
        #Assuming surrogate has registered we
        x=x.to(self.surrogate.reg.device)
        y=y.type(self.surrogate.reg.dtype).to(self.surrogate.reg.device)
        
        nobs=y.shape[-1];
        
        ws=self.surrogate.regress(x,y);
        best_scores=[];
        current_best=(y-target.view(1,-1)).abs().mean(dim=1).max();
        
        #Tool for checking what options are available for tokens[ind]
        #Tokens: 1D tensor
        #ind: int
        #dupes: 2D tensor
        #valid_tokens: list
        def check_dupe(tokens,ind,dupes,valid_tokens):
            l=len(tokens);
            q=tokens.clone().to(dupes.device);
            q[ind]=-1;
            #Identify available indicies
            available_tokens=copy.deepcopy(valid_tokens);
            if len(dupes)>0:
                match=(dupes-q.view(1,-1)).eq(0).long().sum(dim=1).ge(l-1).nonzero();
                match=match.view(-1).tolist()
                if len(match)>0:
                    dupe_tokens=[int(dupes[j][ind]) for j in match];
                    available_tokens=list(set(valid_tokens).difference(set(dupe_tokens)));
            
            return available_tokens;
        
        
        with torch.no_grad():
            tokens=torch.LongTensor(self.surrogate.maxl).fill_(self.pad);
            candidates=[];
            for i in range(1,l+1):
                best_score=1e10;
                tokens[0:i]=torch.LongTensor(self.generate_random_sequence(i));
                while True:
                    improved=False
                    for ind in range(0,i):
                        #Generate input with options
                        available_tokens=check_dupe(tokens,ind,x,self.valid_tokens);
                        tokens_=tokens.tolist();
                        tokens_[ind]=torch.LongTensor(available_tokens).cuda();
                        
                        #Evaluate all options
                        qy,qy_std=self.surrogate.score(tokens_,*ws);
                        #qy_std=qy_std*s_cal;
                        
                        qy_avg=(qy-target.view(1,-1)).abs().mean(dim=-1);
                        qy_std_avg=(qy_std**2).sum(dim=-1);
                        qy_std_avg=qy_std_avg**0.5/qy.shape[-1];
                        
                        #Compute expected improvement
                        #This is minimization version
                        from scipy.stats import norm
                        imp=qy_avg-current_best
                        z=imp/(qy_std_avg+1e-8);
                        z=z.double()
                        
                        cdfz=0.5*(1+torch.erf(z/math.sqrt(2)));
                        pdfz=1/math.sqrt(2*math.pi)*torch.exp(-0.5*z**2);
                        
                        ei=imp.double()*cdfz+qy_std_avg.double()*pdfz
                        
                        #Find largest expected improvement
                        s=-qy_std_avg;
                        score,j=s.min(dim=0);
                        score=float(score);
                        
                        j=int(tokens_[ind][j]);
                        if score<best_score:
                            best_score=score
                            tokens[ind]=j;
                            improved=True;
                    
                    if not improved:
                        break;
                
                candidates.append(tokens.tolist());
                best_scores.append(float(score));
        
        return candidates,best_scores
    



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
        
        self.pre=False;
        self.embeddings=[];
        
        return;
    
    def forward(self,x):
        if isinstance(x,list):
            #Use precalculated embedding lookup
            e=[];
            for i in range(len(x)):
                if isinstance(x[i],int):
                    e_i=self.embedding[i][x[i]:x[i]+1,:];
                else:
                    e_i=self.embedding[i][x[i].view(-1),:];
                e.append(e_i);
            
            #Add up the embeddings
            e_=e[0];
            for i in range(1,len(x)):
                e_=e_+e[i];
            
            e_=e_+self.layers[0].bias.data.view(1,-1);
            h=e_;
            if len(self.layers)>=2:
                h=F.relu(h);
                for i in range(1,len(self.layers)-1):
                    h=self.layers[i](h);
                    h=F.relu(h);
                    #h=F.dropout(h,training=self.training);
                
                h=self.layers[-1](h);
            
            return h
        
        else:
            h=x.view(-1,self.ninput);
            #h=self.bn(h);
            for i in range(len(self.layers)-1):
                h=self.layers[i](h);
                h=F.relu(h);
                #h=F.dropout(h,training=self.training);
            
            h=self.layers[-1](h);
            h=h.view(*(list(x.shape[:-1])+[-1]));
        
        return h
    
    def pre_multiply(self,we):
        nh=we.shape[1];
        
        #Check how many words are there in the input
        n=self.layers[0].weight.shape[1]//nh;
        
        #Convert layer 0 into embeddings
        self.pre=True;
        self.embedding=[];
        for i in range(n):
            e=torch.mm(we,self.layers[0].weight.data[:,i*nh:(i+1)*nh].t());
            self.embedding.append(e.data);
        
        return;

def kernel(x,x2):
    k=2;
    nh=x.shape[-1]//(k*2);
    x=x.view(*(x.shape[:-1]),k,2,nh);
    x2=x2.view(*(x2.shape[:-1]),k,2,nh);
    
    x_=x.transpose(-3,-4).contiguous();
    x2_=x2.transpose(-3,-4).contiguous();
    #print(x.shape,x2.shape,x.select(-2,0).shape,x2.select(-2,0).shape)
    
    K_a=torch.matmul(x_.select(-2,0),x2_.select(-2,0).transpose(-1,-2));
    K_b=torch.matmul(x_.select(-2,1),x2_.select(-2,1).transpose(-1,-2));
    
    
    
    
    K=(K_a*K_b).sum(-3);
    return K

def kernel_z(x):
    k=2;
    nh=x.shape[-1]//(k*2);
    x=x.view(*(x.shape[:-1]),k,2,nh);
    x=(x**2).sum(-1);
    
    Kz=x.select(-1,0)*x.select(-1,1);
    Kz=Kz.sum(dim=-1,keepdim=True);
    return Kz;



def ridge_learn(X,Y,reg):
    N=X.shape[-2];
    nh=X.shape[-1];
    
    b=Y.mean(-2,keepdim=True); #b1y
    
    K=kernel(X,X); #bnn
    if len(X.shape)>=3:
        A=K+reg*torch.eye(N).unsqueeze(0).to(X.device)
    else:
        A=K+reg*torch.eye(N).to(X.device)
    
    A=torch.inverse(A.double()).type(reg.data.dtype);
    a=torch.matmul(A,Y-b);
    
    return b,X,A,a

def ridge_predict(X2,b,X,A,a):
    
    K2=kernel(X2,X);
    Kz=kernel_z(X2)
    
    qy=torch.matmul(K2,a)+b;
    
    qy_var=Kz-(torch.matmul(K2,A)*K2).sum(dim=-1,keepdim=True);
    qy_var[qy_var.lt(0)]=1e-20;
    qy_std=qy_var**0.5;
    
    sz=[1 for i in qy.shape];
    sz[-1]=qy.shape[-1];
    return qy,qy_std.repeat(*sz);
    







class surrogate(nn.Module):
    def __init__(self,we,maxl=8):
        super(surrogate,self).__init__()
        we=we.data.clone().float();
        nhword=we.shape[1];
        
        nh=512*2;
        nh2=512;
        nlayers=3;
        
        self.we=nn.Parameter(we);
        self.encoder=MLP(nhword*maxl,nh2,nh,nlayers);
        self.reg=nn.Parameter(torch.Tensor(1).fill_(0));
        
        self.nhword=nhword;
        self.nh=nh;
        self.nlayers=nlayers;
        self.maxl=maxl;
        return;
    
    def register(self,*args,**kwargs):
        self.encoder.pre_multiply(self.we.data);
    
    def embed(self,x):
        if isinstance(x,list):
            return self.encoder(x);
        else:
            e=self.we[x.view(-1),:];
            e=e.view(*(list(x.shape)[:-1]+[-1]));
            e=self.encoder(e)
            #e=F.normalize(e,dim=-1)
            return e;
        
        return e;
    
    #Kernel GP regression
    #x: n x k, y: n x 1
    #Output embedding e for test kernel, a for kernel weight, b for classifier bias, A for uncertainty calcs
    def regress(self,x,y):
        reg=torch.exp(10*self.reg)+1e-8;
        e=self.embed(x); #Nxnh
        w0=ridge_learn(e,y,reg);
        return (w0,);
    
    
    def score(self,qx,w0):
        e=self.embed(qx);
        qy,qy_std=ridge_predict(e,*w0); #ny
        return qy,qy_std;
    
    def forward(self,x,y,qx):
        ws=self.regress(x,y);
        qy,qy_std=self.score(qx,*ws);
        return qy;
    
    
