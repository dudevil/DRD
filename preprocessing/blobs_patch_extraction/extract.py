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
import skimage as ski
import pandas as pd
import os

PATCH_RADIUS = 50

# In[] load and filter dataset

path = 'D:/projects/kaggle-retina-diabetic/data/train_1024'

df = pd.DataFrame.from_csv('blobs_1024_labeled.tsv', header=-1, index_col=None)
df.columns = ['name','x','y','sigma','strength','tr','lbl']

# In[]
df_dark = df[ (df['lbl'] < 3.)
            & (df['tr'] < 0.) 
            & (df['strength'] > 450.) 
            & ( ((df['x'] - 512)**2 + (df['y'] - 512)**2)**0.5 < 512 ) ]
N = 6
df_part = df_dark.iloc[np.random.choice(xrange(len(df_dark)), N**2)]

#



# In[]


distribution_012 = [len(df[ (df['lbl'] < 3.)
            & (df['tr'] < 0.) 
            & (df['strength'] > thresh) 
            & ( ((df['x'] - 512)**2 + (df['y'] - 512)**2)**0.5 < 512 ) ]) for thresh in np.linspace(300, 900, 25)]
            
distribution_12 = [len(df[ (df['lbl'] < 3.) & (df['lbl'] > 0.)
            & (df['tr'] < 0.) 
            & (df['strength'] > thresh) 
            & ( ((df['x'] - 512)**2 + (df['y'] - 512)**2)**0.5 < 512 ) ]) for thresh in np.linspace(300, 900, 25)]

# In[]
line_012, = plt.plot(np.linspace(300, 900, 25), distribution_012, label='classes (0, 1, 2)')
line_12, = plt.plot(np.linspace(300, 900, 25), distribution_12, label='classes (1, 2)')
plt.legend(handles=[line_012, line_12])
plt.xticks(np.linspace(300, 900, 25))
plt.yticks(np.linspace(0, 2000000, 21))
plt.title("Amount of dark blobs")
plt.xlabel('Threshold')
plt.grid(True)

# In[] Sandbox

def get_patch(i):
    blob = df_part.iloc[i]  
    img = imread(os.path.join(path, blob['name'] + '.png'))  
    sz = PATCH_RADIUS
    patch = img[blob['y']-sz:blob['y']+sz, blob['x']-sz:blob['x']+sz]
    return patch

patches = [get_patch(i) for i in range(len(df_part))]

# In[]
fig, axes = plt.subplots(nrows=N, ncols=N)

for i in range(len(patches)):
    ax = axes[i % N, i / N]
    ax.imshow(patches[i])
    ax.axis('off')
    #ax.set_title(, fontsize=20)

