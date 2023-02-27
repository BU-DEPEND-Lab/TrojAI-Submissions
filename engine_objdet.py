
import cv2
import os
import numpy as np
import json
import copy
import torch
import torch.nn.functional as F
import torchvision
from torchvision.io import read_image
import util.db as db
from PIL import Image

def prepare_boxes(anns, image_id):
    if len(anns) > 0:
        boxes = []
        class_ids = []
        for answer in anns:
            boxes.append(answer['bbox'])
            class_ids.append(answer['category_id'])

        class_ids = np.stack(class_ids)
        boxes = np.stack(boxes)
        # convert [x,y,w,h] to [x1, y1, x2, y2]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    else:
        class_ids = np.zeros((0))
        boxes = np.zeros((0, 4))

    degenerate_boxes = (boxes[:, 2:] - boxes[:, :2]) < 8
    degenerate_boxes = np.sum(degenerate_boxes, axis=1)
    if degenerate_boxes.any():
        boxes = boxes[degenerate_boxes == 0, :]
        class_ids = class_ids[degenerate_boxes == 0]
    target = {}
    target['boxes'] = torch.as_tensor(boxes)
    target['labels'] = torch.as_tensor(class_ids).type(torch.int64)
    target['image_id'] = torch.as_tensor(image_id)
    return target



class new:
    def __init__(self,model_filepath):
        self.model_filepath = model_filepath
        self.model=torch.load(model_filepath).cuda();
        print(type(self.model))
        #self.model.half();
        self.model.eval()

    def load_examples(self,examples_dirpath=None):
        if examples_dirpath is None:
            examples_dirpath = os.path.join(os.path.dirname(self.model_filepath), 'clean-example-data/')
        
        
        fvs=[]
        labels=[];
        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".jpg"):
                #print(examples_dir_entry.path, examples_dir_entry.path.split('.jpg')[0])
                feature_vector = read_image(examples_dir_entry.path).unsqueeze(dim = 0).float()
                 
                fvs.append(feature_vector)

                
                ground_truth_filepath = examples_dir_entry.path.split('.jpg')[0] + '.json'  
                #print(ground_truth_filepath)
                with open(ground_truth_filepath, 'r') as ground_truth_file:
                    ground_truth = ground_truth_file.readline()
                labels.append(torch.tensor([[int(ground_truth)]]));

 
        fvs=torch.cat(fvs,dim=0);
        labels = torch.cat(labels, dim = 0);
        return {'fvs':fvs,'labels':labels};
    #Load data into memory, for faster processing.
    
    def load_examples_w_annotations(self,examples_dirpath = None,scratch_dirpath='',bsz=12,shuffle=False):
        if examples_dirpath is None:
            examples_dirpath = os.path.join(os.path.dirname(self.model_filepath), 'clean-example-data/')
        fns=[os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.jpg')]
        fns.sort()
        
        images=[];
        image_paths=[];
        targets=[];
        annotations=[];
        
        for fn in fns[:bsz]:
            image_id = os.path.basename(fn)
            image_id = int(image_id.replace('.jpg',''))
            # load the example image
            image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)  # loads to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB

            # load the annotation
            with open(fn.replace('.jpg', '.json')) as json_file:
                # contains a list of coco annotation dicts
                annotations = json.load(json_file)

            with torch.no_grad():
                # convert the image to a tensor
                # should be uint8 type, the conversion to float is handled later
                image = torch.as_tensor(image)
                # move channels first
                image = image.permute((2, 0, 1))
                # convert to float (which normalizes the values)
                image = torchvision.transforms.functional.convert_image_dtype(image, torch.float)
                
                target = prepare_boxes(annotations, image_id)
                
                image_paths.append(fn);
                images.append(image)  # wrap into list
                targets.append(target)
                annotations.append(annotations)
        
        return db.Table({'imname':image_paths,'im':images,'target':targets,'annotation':annotations});
    
    #bbox: xywh,relative position to border
    def insert_trigger(self,examples,trigger,bbox,eps=1e-8):
        N=len(bbox);
        x,y,w,h=bbox[:,0],bbox[:,1],bbox[:,2],bbox[:,3]
        
        sx=1/(w+eps);
        sy=1/(h+eps);
        tx=-(2*x-1)*sx-1;
        ty=-(2*y-1)*sy-1;
        
        T=torch.stack((sx,x*0,tx,y*0,sy,ty),dim=-1).view(N,2,3).cuda();
        
        triggered_examples=db.Table(copy.deepcopy(examples.d));
        triggered_examples['im']=[];#triggered_examples['im'].clone()
        for i in range(len(examples)):
            im=examples['im'][i].unsqueeze(0)
            grid=F.affine_grid(T[i:i+1],im.shape)
            overlay=F.grid_sample(trigger.unsqueeze(0).cuda(),grid);
            
            alpha=overlay[:,3,:,:];
            color=overlay[:,:3,:,:];
            
            im_out=im.cuda()*(1-alpha)+color*alpha;
            triggered_examples['im'].append(im_out.squeeze(0))
            
        
        return triggered_examples;
    
    #Perform inference
    #Compute 
    def inference(self,examples):
        #with torch.no_grad():
        scores =self.model(examples.cuda())
        _, preds = scores.max(dim = 0)
        return scores, preds
    
    

