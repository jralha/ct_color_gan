#%% TEST RUN

import cv2
import numpy as np
import argparse
import glob
from tqdm.auto import tqdm
import os
from pathlib import Path

#%%
def bkg_fix(imgdir):

    img_folder = Path(imgdir)
    out_folder = str(img_folder.parent)+'\\bg_fix'
    img_folder = str(img_folder)

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    img_list = glob.glob(img_folder+'\\'+'*.png')

    i = 0
    for img in tqdm(img_list):
        if 'real' in img:
            real = cv2.imread(img)
            fake = cv2.imread(img_list[i-1])
            fake = cv2.cvtColor(fake,cv2.COLOR_BGR2BGRA)

            real_bw = cv2.cvtColor(real,cv2.COLOR_BGR2GRAY)

            bg_bin = cv2.threshold(real_bw,0,255,cv2.THRESH_BINARY)[1]
            bg_bin = cv2.cvtColor(bg_bin,cv2.COLOR_GRAY2BGRA)//255

            bg_fix = bg_bin*fake
            alpha = bg_bin[:,:,0]*255

            bg_fix[:,:,3]=alpha

            outname = img.split('\\')[-1].split('_')[0]
            out = out_folder+'\\'+outname+'.png'

            outimg = cv2.imwrite(out,bg_fix)
            if outimg != True:
                raise Exception("Could not write image")
        i+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgdir',required=True)
    args = parser.parse_args()

    imgdir = args.imgdir

    bkg_fix(imgdir)
# %%
