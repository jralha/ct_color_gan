#%%
import os
import glob
from tqdm.auto import tqdm
from nii_3d import load_split_nii
import sys
import cv2

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

#%%
nii_list = glob.glob('nii_data\\*Energy1.nii')

# %%
num_files=[]
thresh=0 
if __name__ == '__main__':
    for nii in tqdm(nii_list):
        outname = nii.split('\\')[-1].split('.')[0]
        name = nii.split('\\')[-1].split('.')[0]
        outfolder = 'temp_data\\'+name
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
            load_split_nii(nii,outfolder,1,thresh)
        num = len(glob.glob(outfolder+'\\*.png'))
        num_files.append(num)

        results_dir = '.\\results\\'+name
        dataroot = '.\\temp_data\\'+name
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
            
        #######################################################################

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
        # create a website
        web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        if opt.load_iter > 0:  # load_iter is 0 by default
            web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
        print('creating web directory', web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
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



        #######################################################################



        img_folder = '.\\results\\'+name+'\\ct_color\\test_latest\\images'


        bgdir = '.\\results\\'+name+'\\ct_color\\bg_fix'
        if not os.path.exists(bgdir):
            os.mkdir(bgdir)

        img_list = glob.glob(img_folder+'\\'+'*.png')

        i = 0
        for img in (img_list):
            if 'real' in img:
                real = cv2.imread(img)
                fake = cv2.imread(img_list[i-1])

                real_bw = cv2.cvtColor(real,cv2.COLOR_RGB2GRAY)

                bg_bin = cv2.threshold(real_bw,0,255,cv2.THRESH_BINARY)[1]
                bg_bin = cv2.cvtColor(bg_bin,cv2.COLOR_GRAY2RGB)//255

                bg_fix = bg_bin*fake

                outname = img.split('\\')[-1].split('_')[0]
                chars = len(outname)

                if chars == 1:
                    outname = '000'+str(outname)
                elif chars == 2:
                    outname = '00'+str(outname)
                elif chars == 3:
                    outname = '0'+str(outname)


                out = bgdir+'\\'+outname+'.png'

                cv2.imwrite(out,bg_fix)
            i=i+1

        sys.argv = bkp_argv

        #test

# %%
