#Python2,3 compatible headers
from __future__ import unicode_literals,division
from builtins import int
from builtins import range

#System packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import math
import time
import random
import argparse
import sys
import os
import re
import copy
import importlib
from collections import namedtuple
from collections import OrderedDict
from itertools import chain

import util.db as db
import util.smartparse as smartparse
import util.file
import util.session_manager as session_manager
import dataloader

import sklearn.metrics
from hyperopt import hp, tpe, fmin
import json

# Training settings
def default_params():
    params=smartparse.obj();
    #Data
    params.nsplits=4;
    params.pct=0.5
    #Model
    params.arch='arch.mlp_eig';
    params.data='data_r8_eig.pt';
    params.tag='';
    #MISC
    params.session_dir=None;
    params.budget=100;
    return params

def create_session(params):
    session=session_manager.Session(session_dir=params.session_dir); #Create session
    torch.save({'params':params},session.file('params.pt'));
    pmvs=vars(params);
    pmvs=dict([(k,pmvs[k]) for k in pmvs if not(k=='stuff')]);
    print(pmvs);
    util.file.write_json(session.file('params.json'),pmvs); #Write a human-readable parameter json
    session.file('model','dummy');
    return session;


params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;

data=dataloader.new(params.data);
data.cuda();
params.stuff=data.preprocess();

import pandas
meta=pandas.read_csv('/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/METADATA.csv');
meta_table={};
meta_table['model_name']=list(meta['model_name']);
meta_table['label']=[int(x) for x in list(meta['poisoned'])];
meta_table['model_architecture']=list(meta['model_architecture']);
#meta_table['task_type']=list(meta['task_type']);

#meta_table['trigger_option']=list(meta['trigger_option']);
#meta_table['trigger_type']=list(meta['trigger_type']);
meta_table=db.Table(meta_table)
assert 'model_name' in data.data['table_ann'].d.keys()


data.data['table_ann']=db.left_join(data.data['table_ann'],meta_table,'model_name');

for k in data.data['table_ann'].d.keys():
    if isinstance(data.data['table_ann'][k],list):
        if len(data.data['table_ann'][k])>0 and torch.is_tensor(data.data['table_ann'][k][0]):
            print('sending to cuda')
            for i in range(len(data.data['table_ann'][k])):
                data.data['table_ann'][k][i]=data.data['table_ann'][k][i].cuda();



#precompute ws
arch=importlib.import_module(params.arch);
session=create_session(params);
params.session=session;

#Hyperparam search config
hp_config=[];

#   Architectures
#archs=['arch.mlpv2','arch.mlpv3','arch.mlpv4','arch.mlpv5','arch.mlpv6'];
archs=[params.arch];

hp_config.append(hp.choice('arch',archs));
hp_config.append(hp.qloguniform('nh',low=math.log(16),high=math.log(512),q=1));
hp_config.append(hp.qloguniform('nh2',low=math.log(16),high=math.log(512),q=1));
hp_config.append(hp.qloguniform('nh3',low=math.log(16),high=math.log(512),q=1));
hp_config.append(hp.quniform('nlayers',low=1,high=12,q=1));
hp_config.append(hp.quniform('nlayers2',low=1,high=12,q=1));
hp_config.append(hp.quniform('nlayers3',low=1,high=12,q=1));
hp_config.append(hp.loguniform('margin',low=math.log(2),high=math.log(1e1)));
#   OPT
hp_config.append(hp.qloguniform('epochs',low=math.log(10),high=math.log(1000),q=1));
hp_config.append(hp.loguniform('lr',low=math.log(1e-6),high=math.log(1e-2)));
hp_config.append(hp.loguniform('decay',low=math.log(1e-8),high=math.log(1e-3)));
hp_config.append(hp.qloguniform('batch',low=math.log(8),high=math.log(64),q=1));

#Function to compute performance
def configure_pipeline(params,arch,nh,nh2,nh3,nlayers,nlayers2,nlayers3,margin,epochs,lr,decay,batch):
    params_=smartparse.obj();
    params_.arch=arch;
    params_.nh=int(nh);
    params_.nh2=int(nh2);
    params_.nh3=int(nh3);
    params_.nlayers=int(nlayers);
    params_.nlayers2=int(nlayers2);
    params_.nlayers3=int(nlayers3);
    params_.margin=margin;
    params_.epochs=int(epochs);
    params_.lr=lr;
    params_.decay=decay;
    params_.batch=288#int(batch);
    params_=smartparse.merge(params_,params);
    return params_;

crossval_splits=[];
folds=data.generate_random_crossval_folds(nfolds=params.nsplits);
crossval_splits=[(data_train,data_test,data_test) for data_train,data_test in folds]


best_loss_so_far=1e10;
best_auc_so_far = 0.0;
def run_crossval(p):
    global best_loss_so_far, best_auc_so_far
    max_batch=16;
    arch,nh,nh2,nh3,nlayers,nlayers2,nlayers3,margin,epochs,lr,decay,batch=p;
    params_=configure_pipeline(params,arch,nh,nh2,nh3,nlayers,nlayers2,nlayers3,margin,epochs,lr,decay,batch);
    #print('params', params_)
    test12 = vars(params_)
    #for p in test12:
    #    print(p, test12[p])
    arch_=importlib.import_module(params_.arch);
    #Random splits N times
    auc=[];
    ce=[];
    cepre=[];
    results_by_key={};
    t0=time.time();
    ensemble=[];
    mistakes=[];
    criterion = nn.BCELoss()
    for split_id,split in enumerate(crossval_splits):
        data_train,data_val,data_test=split;
        net=arch_.new(params_, input_size=20, output_size=1).cuda();
        opt=optim.Adam(net.parameters(),lr=params_.lr); #params_.lr

        #Train loop
        best_loss=-1e10;
        best_auc =-1e10;
        best_net=copy.deepcopy(net);

        #Training
        for iter in range(params_.epochs):
            net.train();
            loss_total=[];
            for data_batch in data_train.batches(params_.batch,shuffle=True,full=True):
                #print('print', data_batch)
                opt.zero_grad();
                net.zero_grad();
                data_batch.cuda();
                C=data_batch['label'];
                data_batch.delete_column('label');
                scores_i=net(data_batch);

                # loss=F.binary_cross_entropy_with_logits(scores_i,C.float());
                loss = criterion(scores_i, C.float().view(-1, 1))
                # penny: spos is the mean of the scores of the correct label?
                # spos=scores_i.gather(1,C.view(-1,1)).mean();
                # penny: sneg is the mean of the exponential of the logits?
                # sneg=torch.exp(scores_i).mean();
                # loss=-(spos-sneg+1);
                #print(float(loss))
                l2=0;
                for p in net.parameters():
                    l2=l2+(p**2).sum();

                loss=loss+l2*params_.decay;
                loss.backward();
                loss_total.append(float(loss));
                opt.step();

            loss_total=sum(loss_total)/len(loss_total);

            #Eval every epoch
            net.eval();
            scores=[];
            gt=[]
            for data_batch in data_val.batches(max_batch):
                data_batch.cuda();

                C=data_batch['label'];
                data_batch.delete_column('label');
                scores_i=net(data_batch);
                scores.append(scores_i.data.cpu());
                gt.append(C.data.cpu());

            scores=torch.cat(scores,dim=0);
            gt=torch.cat(gt,dim=0);

            auc_i=sklearn.metrics.roc_auc_score(gt.numpy(),scores.numpy());
            loss_i=float(F.binary_cross_entropy(scores,gt.float().view(-1,1)));
            if best_auc < auc_i:
                best_auc = auc_i;
                best_net = copy.deepcopy(net)

            #if best_loss<auc_i:
            #    best_loss=auc_i;
            #    best_net=copy.deepcopy(net);

            #print('train %.4f, loss %.4f, auc %.4f'%(float(loss_total),float(loss_i),float(auc_i)))
            #for g in opt.param_groups:
            #    g['lr'] = g['lr']*0.98

        #Temperature-scaling calibration on val
        # net=best_net;
        # net.eval();
        # scores=[];
        # gt=[];
        # for data_batch in data_val.batches(max_batch):
        #     data_batch.cuda();

        #     C=data_batch['label'];
        #     data_batch.delete_column('label');
        #     scores_i=net(data_batch);
        #     scores.append(scores_i.data);
        #     gt.append(C);

        # scores=torch.cat(scores,dim=0);
        # gt=torch.cat(gt,dim=0);

        # T=torch.Tensor(1).fill_(0).cuda();
        # T.requires_grad_();
        # opt2=optim.Adamax([T],lr=3e-2);
        # for iter in range(500):
        #     opt2.zero_grad();
        #     loss=F.binary_cross_entropy(scores*torch.exp(-T),gt.float().view(-1,1));
        #     loss.backward();
        #     opt2.step();


        #Eval
        net.eval();
        scores=[];
        scores_pre=[];
        gt=[]
        for data_batch in data_test.batches(max_batch):
            data_batch.cuda();

            C=data_batch['label'];
            data_batch.delete_column('label');
            scores_i=net(data_batch);

            scores.append(scores_i.data.cpu());
            scores_pre.append(scores_i.data.cpu());

            gt.append(C.data.cpu());

        scores=torch.cat(scores,dim=0);
        scores_pre=torch.cat(scores_pre,dim=0);
        gt=torch.cat(gt,dim=0);

        def compute_metrics(scores,gt,keys=None):
            if not keys is None:
                #slicing
                categories=set([k for k in keys if not (k is None or k=='None')]);
                results={};
                for c in categories:
                    ind=[i for i,k in enumerate(keys) if k==c or (k is None or k=='None')];
                    scores_c=[scores[i] for i in ind];
                    gt_c=[gt[i] for i in ind];
                    auc_c,ce_c=compute_metrics(scores_c,gt_c);
                    results[c]={'auc':auc_c,'ce':ce_c};

                return results;
            else:
                #Overall
                auc=float(sklearn.metrics.roc_auc_score(torch.LongTensor(gt).numpy(),torch.Tensor(scores).numpy()));
                ce=float(F.binary_cross_entropy(torch.Tensor(scores),torch.Tensor(gt).view(-1,1)));
                return auc,ce;

        auc_i,ce_i=compute_metrics(scores.tolist(),gt.tolist());
        _,ce_pre_i=compute_metrics(scores_pre.tolist(),gt.tolist());
        for i in range(len(gt)):
            if int(gt[i])==1 and float(scores[i])<=0:
                mistakes.append(params.nsplits*i+split_id);


        for result_key in []:#'task_type','trigger_option','trigger_type']:
            results_i=compute_metrics(scores.tolist(),gt.tolist(),data_test.data['table_ann'][result_key])
            for k in results_i:
                if not k in results_by_key:
                    results_by_key[k]={'auc':[],'ce':[]};
                results_by_key[k]['auc'].append(results_i[k]['auc'])
                results_by_key[k]['ce'].append(results_i[k]['ce']);

        auc.append(auc_i);
        ce.append(ce_i);
        cepre.append(ce_pre_i);
        session.log('Split %d, loss %.4f (%.4f), auc %.4f, time %f'%(split_id,ce_i,ce_pre_i,auc_i,time.time()-t0));

        ensemble.append({'net':net.cpu().state_dict(),'params':params_})

    mistakes=sorted(mistakes);
    session.log('Mistakes: '+','.join(['%d'%i for i in mistakes]));
    auc=torch.Tensor(auc);
    ce=torch.Tensor(ce);
    cepre=torch.Tensor(cepre);

    if float(cepre.mean())<best_loss_so_far:
        best_loss_so_far=float(cepre.mean());
        torch.save(ensemble,session.file('min_loss_model.pt'))
    if float(auc.mean()) >best_auc_so_far:
        best_auc_so_far=float(auc.mean())
        torch.save(ensemble,session.file('max_auc_model.pt'))

    session.log('AUC: %f + %f, CE: %f + %f, CEpre: %f + %f (%s (%d,%d,%d), epochs %d, batch %d, lr %f, decay %f)'%(auc.mean(),2*auc.std(),ce.mean(),2*ce.std(),cepre.mean(),2*cepre.std(),arch,nlayers,nlayers2,nh,epochs,batch,lr,decay));

    #goal=-float(auc.mean());#-2*auc.std()
    goal=float(cepre.mean());

    for k in results_by_key:
        auc=torch.Tensor(results_by_key[k]['auc']);
        ce=torch.Tensor(results_by_key[k]['ce']);
        session.log('\t KEY %s, AUC: %f + %f, CE: %f + %f'%(k,auc.mean(),2*auc.std(),ce.mean(),2*ce.std()));

    return goal;



#Get results from hyper parameter search
best=fmin(run_crossval,hp_config,algo=tpe.suggest,max_evals=params.budget)
if len(best) == 0:
    best=util.macro.obj(best);
params_=configure_pipeline(params, **best);
try:
    hyper_params_str=json.dumps(best);
    session.log('Best hyperparam (%s)'%(hyper_params_str));
except:
    session.log('Best hyperparam (%s)'%params)


#Load extracted features
#fvs_0=torch.load('fvs.pt');
#fvs_1=torch.load('fvs_1.pt');
#fvs=db.union(db.Table.from_rows(fvs_0),db.Table.from_rows(fvs_1));
#fvs.add_index('model_id');

#Load labels
#label=[];
#for i in range(200):
#    fname='/work/projects/trojai-example/data/trojai-round0-dataset/id-%08d/ground_truth.csv'%i;
#    f=open(fname,'r');
#    for line in f:
#        line.rstrip('\n').rstrip('\r')
#        label.append(int(line));
#        break;
#
#    f.close();

#fvs['label']=label;
#data=db.DB({'table_ann':fvs});
#data.save('data.pt');
