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
        # parse each conv filter
        fvs=[]
        for i in range(param.shape[-2]):
            for j in range(param.shape[-1]):
                fv=analyze(param[:,:,i,j], k)
                fvs=fvs+fv;

        return fvs
    elif len(param.shape) == 2:
        norm = torch.linalg.norm(param)
        return [norm]
    else:
        return []


def run(interface, k):
    fvs=[];
    for param in interface.model.parameters():
        fvs=fvs+analyze(param.data, k)

    fvs=torch.stack(fvs)
    return fvs


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
    data=db.DB({'table_ann':data})
    # data_res=db.Table({'model_id':[],'model_name':[],'fvs':[], 'label': []})
    # data_mobilenet=db.Table({'model_id':[],'model_name':[],'fvs':[], 'label': []})
    # data_vt=db.Table({'model_id':[],'model_name':[],'fvs':[], 'label': []})
    # data=db.DB({'mobilenet':data_mobilenet, 'resnet':data_res, 'vt':data_vt})
    
    t0=time.time()
    
    default_params=smartparse.obj();
    default_params.fname='norm_features_150.pt'
    params=smartparse.parse(default_params);
    params.argv=sys.argv
    data.d['params']=db.Table.from_rows([vars(params)]);
    
    meta = pandas.read_csv('/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/METADATA.csv')
    poisoned = {}
    for i in range(len(meta['model_name'])):
        poisoned[meta['model_name'][i]] = meta['poisoned'][i]
            
    model_ids = list(range(0,288))
    
    for i,id in enumerate(model_ids):
        print(i,id)
        try:
            fv = extract_fv(id,params=params);
            data['table_ann']['model_name'].append('id-%08d'%id);
            data['table_ann']['model_id'].append(id);
            #data['table_ann']['label'].append(label);
            data['table_ann']['fvs'].append(fv[-150:]);
            data['table_ann']['label'].append(int(poisoned['id-%08d'%id]))
    
            # model = torch.load('/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/models/id-%08d'%i+'/model.pt').cuda()
            # for param in model.named_parameters():
                # if param[0] == 'features.0.0.weight':
                #     print('Mobile Net V2')
                #     data['mobilenet']['fvs'].append(fv);
                #     data['mobilenet']['model_name'].append('id-%08d'%id);
                #     data['mobilenet']['model_id'].append(id);
                #     data['mobilenet']['label'].append(int(poisoned['id-%08d'%id]))
                #     break
                # if param[0] == 'conv1.weight':
                #     print('ResNet')
                #     data['resnet']['fvs'].append(fv);
                #     data['resnet']['model_name'].append('id-%08d'%id);
                #     data['resnet']['model_id'].append(id);
                #     data['resnet']['label'].append(int(poisoned['id-%08d'%id]))
                #     break
                # if param[0] == 'cls_token':
                #     print('Vision Transformer')
                #     data['vt']['fvs'].append(fv);
                #     data['vt']['model_name'].append('id-%08d'%id);
                #     data['vt']['model_id'].append(id);
                #     data['vt']['label'].append(int(poisoned['id-%08d'%id]))
                #     break
    
            print('Model %d(%d), time %f'%(i,id,time.time()-t0));
        except AttributeError as e:
            print(f"Skip loading Model-{id}: {e}")
    
    data.save(params.fname)
