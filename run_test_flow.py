#%%
import os
import glob
from tqdm.auto import tqdm
from split_nii import load_split_nii
from bkg_fix import bkg_fix
import sys
import cv2

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np

#%%
nii_list = glob.glob('datasets\\nii_data_libra\\*Energy1.nii')

# %%
if __name__ == '__main__':
    for nii in tqdm(nii_list):
        outname = nii.split('\\')[-1].split('.')[0]
        name = nii.split('\\')[-1].split('.')[0]
        split_outfolder = '.\\datasets\\ct_data\\'+name
        if not os.path.exists(split_outfolder):
            os.mkdir(split_outfolder)
            load_split_nii(nii,split_outfolder,1,0)
        num = len(glob.glob(split_outfolder+'\\*.png'))

        results_dir = '.\\results\\'+name
        dataroot = split_outfolder
        args= [ '--dataroot',dataroot,
                '--gpu_ids','-1',
                '--name','ct_color',
                '--no_dropout',
                '--preprocess','none',
                '--num_test',str(num),
                '--results_dir',results_dir]

        bkp_argv = sys.argv
        for arg in args:
            sys.argv.append(arg)
            
        opt = TestOptions().parse()  # get test options
        # hard-code some parameters for test
        opt.num_threads = 0   # test code only supports num_threads = 1
        opt.batch_size = 1    # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers

        if opt.eval:
            model.eval()
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()     # get image paths
            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)



        img_folder = '.\\results\\'+name+'\\ct_color\\test_latest\\images'

        bkg_fix(img_folder)

        sys.argv = bkp_argv

        #test

# %%
