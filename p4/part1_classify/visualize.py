import torch
import numpy as np 
import matplotlib.pyplot as plt
from model import Class_Net
from os.path import join, split
from torchvision import utils
import os


def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path)

path = 'visualize'
mkdir(path)

model = Class_Net()
model = torch.nn.DataParallel(model)
cp_dir = 'revised_class_no_sig'
fn = 'best_checkpoint.pth.tar'
checkpoint = torch.load(join(cp_dir, fn))
model.load_state_dict(checkpoint['state_dict'])

for i, module in enumerate(model.module.named_modules()):
    if 'con' in module[0]:
        # print(i, module[1].weight.data.shape)
        weight = module[1].weight.data.detach()
        n,c,h,w = weight.shape
        if c!=1:
            weight = weight.view(n*c,-1,h,w)
        # grid = utils.make_grid(weight, nrow=8, normalize=True, padding=1)
        if n*c<=64:
            nrow = 8
        else:
            nrow = 32
        utils.save_image(weight, join(path, 'test_'+str(i)+'.jpg'), nrow=nrow, normalize=True, padding=1)
        # padding


# print(.features)