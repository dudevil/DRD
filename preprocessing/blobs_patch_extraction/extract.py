# -*- coding: utf-8 -*-
"""
Created on Sat Jul 04 15:36:22 2015

@author: rakhunzy
"""

# In[] imports
import numpy as np
from skimage.io import imread
from skimage.draw import circle
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage as ski
import pandas as pd
import os
from multiprocessing import Pool

PATCH_RADIUS = 8
PATCH_SIZE = PATCH_RADIUS*2
blob_cols = 0
arr = np.empty((1,1), dtype=object)

class Doer(object):
    def __init__(self, image, group, base):    
        self.image = image
        self.group = group
        self.base = base                 
        
    def __call__(self, index):
        self.process_row(self.image, self.group, self.base, index)  
        return 1
        
    def process_row(self, image, group, base_ind, j):
        img_h, img_w  = self.image.shape
        blob = self.group.irow(j)
        x, y = self.group.iloc[j, 1] - PATCH_RADIUS, self.group.iloc[j, 2] - PATCH_RADIUS
        x_end, y_end = x + PATCH_SIZE, y + PATCH_SIZE
       
        if x > 0 and y > 0 and x_end <= img_h and y_end <= img_w:
            arr[base_ind + j, :blob_cols] = blob
            arr[base_ind + j, blob_cols:] = self.image[y:y_end, x:x_end].reshape(PATCH_SIZE**2)
        else:
            arr[base_ind + j, :blob_cols] = blob
            arr[base_ind + j, blob_cols:] = np.zeros((PATCH_SIZE**2))
        return 1


if __name__ == "__main__":
    print('Reading data')
# In[] load and filter dataset
    data_dir = 'D:/projects/kaggle-retina-diabetic/data/train_1024'
    FILE = 'blobs_1024_labeled.tsv'
    
    dfr = pd.read_csv(FILE, header=-1)
    dfr.columns = ['name','x','y','sigma','strength','tr','lbl']
    
    dfr_work = dfr[ (dfr['lbl'] < 3.)
                  & (dfr['tr'] < 0.)
                  & (dfr['strength'] > 350.) 
                  & ( ((dfr['x'] - 512)**2 + (dfr['y'] - 512)**2)**0.5 < 512 ) ]
    
    rows_count =  len(dfr_work)
                
    test_images = np.unique(dfr_work['name'])
    dfr_work = dfr_work[dfr_work.name.isin(test_images)]
    
    
    dfr_work = dfr_work#[dfr_work['strength'] > 900]
    blob_cols = len(dfr_work.columns)
    arr = np.empty((len(dfr_work), blob_cols + PATCH_SIZE**2), dtype=object)

# In[]    
    print('Processing data')
    images_processed = 0
    images_total = len(dfr_work.groupby('name'))
    i = 0

    for fname, group in dfr_work.groupby('name'):
        img = ski.io.imread(os.path.join(data_dir, fname + '.png'), as_grey=True)
        map(Doer(img, group, i), xrange(len(group)))
        i += len(group)
        images_processed += 1
        print('{}/{}'.format(images_processed, images_total))
    
    np.save('patches_indiced', arr)
    np.savetxt('patches_indiced.csv', arr[:2000], delimiter=",", fmt="%s")  
    
#    # In[]
#    N = 13
#    fig, axes = plt.subplots(nrows=N, ncols=N)
#    for i in range(N**2):
#        ax = axes[i % N, i / N]
#        ax.imshow(arr[i].reshape(PATCH_SIZE, PATCH_SIZE), cmap = plt.get_cmap('gray'), vmin = 0, vmax = 1)
#        ax.axis('off')
#        #ax.set_title(, fontsize=20)