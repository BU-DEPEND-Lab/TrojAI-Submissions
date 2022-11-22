
import os
import datasets
import numpy as np
import torch
import json
import jsonschema
import jsonpickle
import warnings

import torchvision
from torchvision import transforms
import torchvision.datasets.folder

warnings.filterwarnings("ignore")

def get_paths(id,root='/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/'):
    id='id-%08d'%id;
    model_filepath=os.path.join(root,'models',id,'model.pt');
    examples_dirpath=os.path.join(root,'models',id,'clean-example-data');
    scratch_dirpath='./scratch'
    
    return model_filepath, scratch_dirpath, examples_dirpath;

def get_root(id,root='/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/'):
    id='id-%08d'%id;
    return os.path.join(root,'models',id);

def load_trigger(id,root='/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/'):
    id='id-%08d'%id;
    imname=os.path.join(root,'models',id,'trigger_0.png');
    im=torchvision.datasets.folder.default_loader(imname);
    transform=transforms.ToTensor()
    trigger=transform(im);
    alpha=(trigger**2).sum(dim=0,keepdim=True).gt(0.01);
    trigger=torch.cat((trigger,alpha),dim=0);
    return trigger;



id2label={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

def visualize(imname,data=None,out='tmp.png',threshold=0.1):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots()
    
    if isinstance(imname,str):
        img = mpimg.imread(imname)
    elif torch.is_tensor(imname):
        img=imname.permute(1,2,0).cpu().numpy();
    else:
        img=transforms.ToTensor()(imname);
        img=img.permute(1,2,0).cpu().numpy();
    
    
    imgplot = ax.imshow(img) #HWC
    
    if not data is None:
        for i in range(len(data['boxes'])):
            bbox=data['boxes'][i]
            #label='L%d'%int(data['labels'][i])#id2label[int(data['labels'][i])];
            label=id2label[int(data['labels'][i])];
            score=float(data['scores'][i]);
            
            if score>threshold:
                x0,y0,x1,y1=int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
                rect=patches.Rectangle((x0,y0),x1-x0,y1-y0, linewidth=3, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                
                t=ax.text(x0,y0,'%s: %.3f'%(label,score),color='white',weight='bold',ha='left',va='top');
                t.set_bbox(dict(facecolor='black', alpha=0.5, edgecolor='black'))
    
    
    plt.savefig(out)
    plt.close();