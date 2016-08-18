
# coding: utf-8

## After training the network, load the weights and apply network to classify each voxel of a test cases
import sys
import pyprind
import os
import nrrd
import numpy as np
import random
import multiprocessing
num_cores = multiprocessing.cpu_count()
USERPATH = os.path.expanduser("~")
print(USERPATH)
import six.moves.cPickle as pickle
from keras.models import model_from_json

# we load the model with the trained weights
model = model_from_json(open('model_architecture.json').read())
model.load_weights('model_weights.h5')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# we load a test case
nrrdData = nrrd.read('Case064.nrrd')
im = nrrdData[0]
# choose the reasonable region of test patch
pick = im[260:412, 163:300, 65:]
print(pick.shape)

def findtips(N, p1, p2, p3):
    '''
    Find the tip in the image by computing testing patches at every voxel position
    TODO: make this method more efficient
    '''
    xmiddle = pick.shape[0]//2
    ymiddle = pick.shape[1]//2
    zmiddle = pick.shape[2]//2

    x0= xmiddle - xmiddle//N
    y0= ymiddle - ymiddle//N
    z0= zmiddle - zmiddle//N

    xe= xmiddle + xmiddle//N
    ye= ymiddle + ymiddle//N
    ze= zmiddle + zmiddle//N

    tips = []
    bar = pyprind.ProgBar(xmiddle//N*2, title='Find_tip', stream=sys.stdout)
    for xi in range(x0, xe-p1):
        for yi in range(y0, ye-p2):
            vols = [pick[xi:xi+p1,yi:yi+p2,zi:zi+p3] for zi in range(z0,ze-p3)]
            # we normalize the data (centered on mean 0 and rescaled in function of the STD)
            volnorm = [ x-np.mean(x) for x in vols]
            volnorm2 = [x/np.std(x) for x in volnorm]
            cube = np.array(volnorm2)
            pro = model.predict_proba(cube, batch_size=32, verbose=False)
            # choose a suitable confidence threshold here, default=0.5
            indices = np.where(pro[:,0] > 0.5)
            # we add the coordinates of the center voxel of the patches that tested positive
            for i in indices[0]:
                # change coord accoring to region picked
                tips.append([xi+p1/2+260,yi+p2/2+163,z0+p3/2+65+i, pro[i,0]])
        bar.update()
    return tips

# find the tips for patches with size 10-10-10
res = findtips(1, 10, 10, 10)
res = np.array(res)
print(res.shape)

## save all tips
f = open('tips.save', 'wb')
np.save(f, res)
f.close

