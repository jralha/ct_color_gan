import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

plt.style.use('dark_background')

parser = argparse.ArgumentParser()
parser.add_argument('--niifile',required=True)
parser.add_argument('--dim',required=True)
parser.add_argument('--slices',required=True)
parser.add_argument('--c_map',default='gray')
args = parser.parse_args()

filename = args.niifile
dim = args.dim
slcs = args.slices.split(',')
c_map = args.c_map

niiarray = np.array(nib.load(filename).get_data())

if dim == 'xy':
    array= niiarray.T
elif dim == 'xz':
    array=niiarray
elif dim == 'yz':
    array=np.swapaxes(niiarray,1,2)

nimg=len(slcs)
nx=np.ceil(np.sqrt(len(slcs)))
ny=nx
cimg=1
plt.figure()
for s in slcs:
    plt.subplot(nx,ny,cimg)
    plt.imshow(array[int(s)],cmap=c_map)
    plt.yticks([])
    plt.xticks([])
    # for spine in plt.gca().spines.values():
    #     spine.set_visible(False)
    # plt.title('Slice '+str(s))
    cimg=cimg+1

plt.tight_layout()
plt.show()