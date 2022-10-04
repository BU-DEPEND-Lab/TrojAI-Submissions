import torch
import helper_r10 as helper
import engine_objdet as engine

#First try to run inference on a model

model_filepath, scratch_dirpath, examples_dirpath=helper.get_paths(2);
trigger=helper.load_trigger(2);
print(trigger.shape)

interface=engine.new(model_filepath);
examples=interface.load_examples('/work2/project/coco2017/trojai_coco',bsz=100) #examples_dirpath
bbox=torch.Tensor([0.1,0.5,0.2,0.2]);
triggered_examples=interface.insert_trigger(examples,trigger,bbox.view(1,4).repeat(len(examples),1));

#helper.visualize(triggered_examples['im'][0],out='tmp.png')
#a=0/0

results=interface.inference(examples);
results2=interface.inference(triggered_examples);

err=torch.stack(([r['loss']['bbox_regression'] for r in results]),dim=0)
err2=torch.stack(([r['loss']['bbox_regression'] for r in results2]),dim=0)

for i in range(len(examples)):
    helper.visualize(examples['imname'][i],results[i]['pred'],'vis/clean%d.png'%i)
    helper.visualize(triggered_examples['im'][i],results2[i]['pred'],'vis/trigger%d.png'%i)




import weight_analysis 

fvs=weight_analysis.run(interface)
