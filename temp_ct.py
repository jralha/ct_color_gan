#%%
import cv2
import numpy as np
import glob
from tqdm.auto import tqdm

#%%
imglist = glob.glob('datasets/nii_data_libra/*/*.png')
outf = 'datasets\\libra\\tempA\\'
threshold = 10

for imgpath in tqdm(imglist):
    img = cv2.imread(imgpath)
    out=cv2.transpose(img)
    out=cv2.flip(out,flipCode=0)
    fname = imgpath.split('\\')[-2]+'_'+imgpath.split('\\')[-1]
    outname = outf+fname
    if np.mean(out) > threshold:
        outimg = cv2.imwrite(outname,out)
        if outimg == False:
            raise Exception("Could not write image")


# %%
