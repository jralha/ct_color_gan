#%%
import os
import nibabel as nib
import numpy as np
import cv2
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
import glob

#%%
def load_split_nii(filename,dst,dim,threshold,outname=None):
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
        
        if outname != None:
            filename = outname+'_'+str(i)+'.png'
        else:
            filename = str(i)+'.png'
        filepath = os.path.abspath(dst)

        if not os.path.exists(filepath):
            os.mkdir(filepath)

        outpath = os.path.join(filepath,filename)

        if np.mean(img) > threshold:
            outimg = cv2.imwrite(outpath,img)
            if outimg == False:
                raise Exception("Could not write image")

#Split all
if __name__ == '__main__':

    nii_list = glob.glob('datasets\\nii_data_libra\\*Energy2.nii')

    THRESHOLD = 100
    for nii in tqdm(nii_list):
        outname = nii.split('\\')[-1].split('.')[0]
        split_outfolder = '.\\libra\\tempA\\'
        if not os.path.exists(split_outfolder):
            os.mkdir(split_outfolder)
        load_split_nii(nii,split_outfolder,1,THRESHOLD,outname=outname)


# %%
