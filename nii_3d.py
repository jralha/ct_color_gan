#%%
import os
import nibabel as nib
import numpy as np
import cv2
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
import glob
import argparse

#%%
def load_split_nii(filename,dst,dim,threshold):
    scaler = MinMaxScaler(feature_range=(0,255))
    niiarray = np.array(nib.load(filename).get_data())
    nii_slices = range(niiarray.shape[dim])
    
    for i in (nii_slices):
    
        if dim == 0:
            temparray = niiarray[i,:,:]
        elif dim == 1:
            temparray = niiarray[:,i,:]
        elif dim == 2:
            temparray = niiarray[:,:,i]
        else:
            print('Invalid dimension')
            break

        temparray = scaler.fit_transform(temparray)
        temparray = temparray.astype(np.float32)
        img = cv2.cvtColor(temparray,cv2.COLOR_GRAY2BGR)
        
        filename = str(i)+'.png'
        filepath = os.path.abspath(dst)

        if not os.path.exists(filepath):
            os.mkdir(filepath)

        outpath = os.path.join(filepath,filename)

        if np.mean(img) > threshold:
            if not cv2.imwrite(outpath,img):
                raise Exception("Could not write image")

#Split all
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--nii',required=True)
    parser.add_argument('--dst',required=True)
    parser.add_argument('--dim',required=True)
    args = parser.parse_args()

    nii = args.nii
    nii_f = nii.split('\\')[-1].split('.')[0]
    dump_folder = args.dst+'\\'+nii_f
    dim = int(args.dim)

    load_split_nii(nii,dump_folder,dim,0)


# %%
