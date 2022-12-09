import os
import sys
import torch
import time
import json
import jsonschema
import jsonpickle
import pandas

import math
from sklearn.cluster import AgglomerativeClustering


import util.db as db
import util.smartparse as smartparse
import helper_r10 as helper
import engine_objdet as engine

def analyze(param, k):
    if len(param.shape)==1:
        fvs=analyze(param.view(1,-1), k)
        return fvs
    if len(param.shape)==4:
        #Conv filters
        fvs=[];
        for i in range(param.shape[-2]):
            for j in range(param.shape[-1]):
                fv=analyze(param[:,:,i,j], k);
                fvs=fvs+fv;

        return fvs
    elif len(param.shape)==2:
        #Create an overarching matrix
        #print('szcap', szcap)
        # nx=param.shape[1]
        # ny=param.shape[0]
        # n=max(nx,ny);
        # n=min(n,szcap)
        # m=min(nx,ny);
        # z=torch.zeros(n,n).to(param.device);
        # z[:min(ny,n),:min(nx,n)]=param.data[:min(ny,n),:min(nx,n)];

#         print('before svd')
        U, S, V = param.svd()
        # S = S / torch.linalg.norm(S)
#         print('after svd')
        # e,_=z.eig();
        # eig_norm = torch.linalg.norm(e, dim=1)
        #e = e/eig_norm

        mean = S.mean(axis=0).unsqueeze(0)
        std = torch.std(S, dim=0, unbiased=False).unsqueeze(0) # use biased std to avoid NaN

#         print('S shape', S.shape)
        S_expanded = torch.cat((S.unsqueeze(dim=1), torch.zeros((len(S), 1)).cuda()), dim=1)
        eig_norm = torch.linalg.norm(S_expanded, dim=1)
        indices_norm_based_desc = torch.argsort(eig_norm)
        top_k_indices = indices_norm_based_desc[-1 * k:]
        top_k_eigen = S[top_k_indices]
        bottom_k_indices = indices_norm_based_desc[:k]
        bottom_k_eigen = S[bottom_k_indices]

        fv=torch.cat((mean, std, top_k_eigen, bottom_k_eigen, torch.linalg.norm(S).unsqueeze(0)),dim=0)

        return [fv];
    else:
        return [];

def run(interface, k):
    fvs=[];
    for param in interface.model.parameters():
        #print('fvs', len(fvs))
        fvs=fvs+analyze(param.data, k);

    #print()
    #print('fvs shape', len(fvs), len(fvs[0]), len(fvs[49]))
    fvs=torch.stack(fvs);
    #print('after torch stack', len(fvs))
    #print()

    # 302 x 4

    min_features = torch.min(fvs, dim=0).values
    max_features = torch.max(fvs, dim=0).values
    quantiles = torch.quantile(fvs, q=torch.Tensor([0.1, 0.9]).cuda(), dim=0)
    # mean_features = torch.mean(fvs, dim=0)
    # std_features = torch.std(fvs, dim=0)
    # fvs = torch.cat((min_features, max_features, ), dim=0)
    fvs = torch.cat((min_features.unsqueeze(0), max_features.unsqueeze(0), quantiles), dim=0)

    return fvs;



#Fuzzing call for TrojAI R9
def extract_fv(id=None, model_filepath=None, scratch_dirpath=None, examples_dirpath=None, params=None):
    t0=time.time();
    default_params=smartparse.obj();
    params = smartparse.merge(params,default_params);

    if not id is None:
        model_filepath, scratch_dirpath, examples_dirpath=helper.get_paths(id, root = '/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/');

    interface=engine.new(model_filepath);
    fvs=run(interface, 1)

    print('Weight analysis done, time %.2f'%(time.time()-t0))
    return fvs

if __name__ == "__main__":
    data=db.Table({'model_id':[],'model_name':[],'fvs':[], 'label': []}); #,'label':[] label currently incorrect
    data=db.DB({'table_ann':data});

    t0=time.time()

    default_params=smartparse.obj();
    default_params.fname='test.pt'
    params=smartparse.parse(default_params);
    params.argv=sys.argv
    data.d['params']=db.Table.from_rows([vars(params)]);


    meta = pandas.read_csv('/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/METADATA.csv')
    poisoned = {}
    for i in range(len(meta['model_name'])):
        poisoned[meta['model_name'][i]] = meta['poisoned'][i]
        

    model_ids=list(range(0,288))

    for i,id in enumerate(model_ids):
        print(i,id)
        try:
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
            data['table_ann']['label'].append(int(poisoned['id-%08d'%id]))

            # print(fv)
            print('Model %d(%d), time %f'%(i,id,time.time()-t0));
            if i%1==0:
                data.save(params.fname);
        except AttributeError as e:
            print(f"Skip loading Model-{id}: {e}")

    print(data['table_ann']['fvs'])
    data.save(params.fname);

