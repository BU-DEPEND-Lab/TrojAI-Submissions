import torch
import torch.nn as nn
import torchvision

def loss(pred, target):
    pred_classes = [item['label'] for item in pred]
    target_class = [item['label'] for label in target]
    loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
    loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
    loss_bb = loss_bb.sum()
    loss = loss_class + loss_bb/C
    return torchvision.ops.generalized_box_iou_loss

def attrition(loss, optimizer, input, model):
    optimizer.zero_grad()
    loss.backward()
    input.grad.d