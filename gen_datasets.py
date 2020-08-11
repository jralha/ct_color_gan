#%%
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os 
from tqdm.auto import tqdm

#%%
def window_image(image,outfolder,window_size=220,vstep=220,hstep=220,split=None,split_folder=None,thresh=0):
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    img = cv2.imread(image)
    x=img.shape[1]
    y=img.shape[0]
    nx = np.floor(x/hstep)
    ny = np.floor(y/vstep)

    if split != None:
        counter = np.floor(1/split)
    else:
        counter=-1

    cy=0
    d=window_size
    n=0
    for i in range(int(ny)):
        cx=0
        for j in range(int(nx)):
            slc = img[cy:cy+d,cx:cx+d]
            if slc.shape[0:2] == (d, d):
                name = str(i)+'_'+str(j)+'_'+image.split('\\')[-1]
                if n != counter:
                    outpath = os.path.join(outfolder,name)
                else:
                    outpath = os.path.join(split_folder,name)
                    n=0
                if np.mean(slc) > thresh:
                    outimg = cv2.imwrite(outpath,slc)
                    if outimg == False:
                        raise Exception("Could not write image")
            n=n+1
            cx = cx+hstep
        cy = cy+vstep


#%%
tempA = glob.glob('datasets\\libra\\tempA\\*.png')
tempB = glob.glob('datasets\\libra\\tempB\\*.png')
outA = ('datasets\\libra\\trainA\\ct','datasets\\libra\\testA\\ct')
outB = ('datasets\\libra\\trainB\\foto','datasets\\libra\\testB\\foto')

#%%
for img in tqdm(tempA):
    window_image(img,outA[0],split=0.2,split_folder=outA[1],thresh=10)

# %%
for img in tqdm(tempB):
    window_image(img,outB[0],split=0.2,split_folder=outB[1],window_size=1000,vstep=800,hstep=800,thresh=100)

# %%
