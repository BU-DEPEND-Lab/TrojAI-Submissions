


import os
import sys
import torch
import time
import json
import jsonschema
import jsonpickle

import math
from sklearn.cluster import AgglomerativeClustering


import util.db as db
import util.smartparse as smartparse
import helper_r10 as helper
import engine_objdet as engine


#value at top x percentile, x in 0~100
def hist_v(w,bins=100):
    s,_=w.sort(dim=0);
    wmin=float(w.min());
    wmax=float(w.max());

    n=s.shape[0];
    hist=torch.Tensor(bins);
    for i in range(bins):
        x=math.floor((n-1)/(bins-1)*i)
        x=max(min(x,n),0);
        v=float(s[x]);
        hist[i]=v;

    return hist;

def analyze(param,nbins=100,szcap=4096):
    if len(param.shape)==1:
        fvs=analyze(param.view(1,-1),nbins=nbins,szcap=szcap)
        return fvs
    if len(param.shape)==4:
        #Conv filters
        fvs=[];
        for i in range(param.shape[-2]):
            for j in range(param.shape[-1]):
                fv=analyze(param[:,:,i,j],nbins=nbins,szcap=szcap);
                fvs=fvs+fv;

        return fvs
    elif len(param.shape)==2:
        #Create an overarching matrix
        nx=param.shape[1]
        ny=param.shape[0]
        n=max(nx,ny);
        n=min(n,szcap)
        m=min(nx,ny);
        z=torch.zeros(n,n).to(param.device);
        z[:min(ny,n),:min(nx,n)]=param.data[:min(ny,n),:min(nx,n)];

        #
        e,_=z.eig();
        e = e/torch.linalg.norm(e)

        #Analyze eigs
        #1) eig distribution
        e2=(e**2).sum(1);
        rank=int(e2.gt(0).long().sum());
        if rank<m:
            #pad 0s to eig
            e_nz=e[e2.gt(0)].clone();
            e_z=torch.Tensor(m-rank,2).fill_(0).to(e.device);
            e=torch.cat((e_nz,e_z),dim=0);
        else:
            #Still adds a 0 for perspective
            e_nz=e[e2.gt(0)].clone();
            e_z=torch.Tensor(1,2).fill_(0).to(e.device);
            e=torch.cat((e_nz,e_z),dim=0);

        #Get histogram of abs, real, imaginary
        e2=(e**2).sum(1)**0.5;
        e2_hist=hist_v(e2,nbins);
        er_hist=hist_v(e[:,0],nbins);
        ec_hist=hist_v(e[:,1],nbins);

        #2) histogram of eig persistence
        cm=AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage='single')
        cm=cm.fit(e.cpu())
        d=torch.from_numpy(cm.distances_);
        eig_persist=hist_v(d,nbins)


        #Get histogram of weight value and abs
        w=param.data.view(-1);
        w_hist=hist_v(w,nbins);
        wabs_hist=hist_v(w.abs(),nbins);

        fv=torch.cat((e2_hist,er_hist,ec_hist,eig_persist,w_hist,wabs_hist),dim=0);
        return [fv];
    else:
        return [];

def run(interface,nbins=100,szcap=4096):
    fvs=[];
    for param in interface.model.parameters():
        fvs=fvs+analyze(param.data,nbins=nbins,szcap=szcap);

    fvs=torch.stack(fvs);
    return fvs;



#Fuzzing call for TrojAI R9
def extract_fv(id=None, model_filepath=None, scratch_dirpath=None, examples_dirpath=None, params=None):
    t0=time.time();
    default_params=smartparse.obj();
    default_params.nbins=100
    default_params.szcap=4096
    params = smartparse.merge(params,default_params);

    if not id is None:
        model_filepath, scratch_dirpath, examples_dirpath=helper.get_paths(id);

    interface=engine.new(model_filepath);
    fvs=run(interface,nbins=params.nbins,szcap=params.szcap)

    print('Weight analysis done, time %.2f'%(time.time()-t0))
    return fvs

if __name__ == "__main__":
    data=db.Table({'model_id':[],'model_name':[],'fvs':[]}); #,'label':[] label currently incorrect
    data=db.DB({'table_ann':data});

    t0=time.time()

    default_params=smartparse.obj();
    default_params.nbins=100;
    default_params.szcap=4096;
    default_params.fname='data_r10_weight.pt'
    params=smartparse.parse(default_params);
    params.argv=sys.argv;
    data.d['params']=db.Table.from_rows([vars(params)]);

    model_ids=list(range(0,144))

    for i,id in enumerate(model_ids):
        print(i,id)
        fv=extract_fv(id,params=params);

        '''
        #Load GT
        fname=os.path.join(helper.get_root(id),'ground_truth.csv');
        f=open(fname,'r');
        for line in f:
            line.rstrip('\n').rstrip('\r')
            label=int(line);
            break;

        f.close();
        '''

        data['table_ann']['model_name'].append('id-%08d'%id);
        data['table_ann']['model_id'].append(id);
        #data['table_ann']['label'].append(label);
        data['table_ann']['fvs'].append(fv);

        print('Model %d(%d), time %f'%(i,id,time.time()-t0));
        if i%1==0:
            data.save(params.fname);

    data.save(params.fname);

