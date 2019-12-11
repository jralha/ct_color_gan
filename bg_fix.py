#%% TEST RUN

import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%


real = cv2.imread('ct_color_gan\\results\\ct_color\\test_latest\\images\\88_real.png')
fake = cv2.imread('ct_color_gan\\results\\ct_color\\test_latest\\images\\88_fake.png')

# real = cv2.cvtColor(real,cv2.COLOR_BGR2RGB)
# fake = cv2.cvtColor(fake,cv2.COLOR_BGR2RGB)

real_bw = cv2.cvtColor(real,cv2.COLOR_RGB2GRAY)

# bg_bin = cv2.threshold(real_bw,0,255,cv2.THRESH_OTSU)[1]
bg_bin = cv2.threshold(real_bw,0,255,cv2.THRESH_BINARY)[1]


bg_bin = cv2.cvtColor(bg_bin,cv2.COLOR_GRAY2RGB)//255
test = bg_bin*fake

plt.figure(figsize=[100,30])
plt.subplot(4,1,1)
plt.imshow(fake)
plt.subplot(4,1,2)
plt.imshow(real)
plt.subplot(4,1,3)
plt.imshow(bg_bin)
plt.subplot(4,1,4)
plt.imshow(test[:,:,[2,1,0]])
# %%
cv2.imwrite('test.png',test)

# %%
