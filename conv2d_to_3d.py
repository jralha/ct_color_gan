#%%
import os
import sys

cdir = os.getcwd()
os.chdir('ct_color_gan')
from models.base_model import BaseModel
from models.test_model import TestModel
from models import networks, create_model
from options.test_options import TestOptions
os.chdir(cdir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib

#%%
nii = np.expand_dims(np.expand_dims(np.uint8(np.array(nib.load('datasets\\nii_data_libra\\Libra_CX1_T1_221_184_1485_Energy1.nii' ).get_data())),axis=0),axis=0)[:,:,:-1,:,:-1]
nii = torch.tensor(nii)

checkpoint = 'ct_color_gan\\checkpoints\\ct_color\\latest_net_G.pth'

#%%
netG = networks.define_G(input_nc=3,output_nc=3,ngf=64,norm='instance',netG='resnet_9blocks',gpu_ids=[])
netG.load_state_dict(torch.load(checkpoint))

# %%
def conv2dto3d(module):
        kernel_size = module.kernel_size[0]
        stride = module.stride[0]
        padding = module.padding[0]
        weight = module.weight.unsqueeze(2) / kernel_size
        weight = torch.cat([weight for _ in range(0, kernel_size)], dim=2)
        bias = module.bias

        module = nn.Conv3d(in_channels=module.weight.shape[1], out_channels=module.weight.shape[0],
                               kernel_size=kernel_size, padding=padding, stride=stride, bias=True)
        module.bias = bias
        return module

modules = {}
for name, module in netG.named_modules():
    if(isinstance(module, nn.Conv2d)):
        module = conv2dto3d(module)

       
    if 'conv_block' in name:
        for name2,module2 in module.named_modules():
            if(isinstance(module2, nn.Conv2d)):
                module2 = conv2dto3d(module)

                print(module2) 
        
# %%
