
# coding: utf-8

# # Find TIPS
# ## After training the network, load the weights and apply network to classify each voxel of a test cases


from __future__ import division

import sys
import os
import nrrd
import numpy as np
import random
import multiprocessing
num_cores = multiprocessing.cpu_count()
USERPATH = os.path.expanduser("~")
print(USERPATH)
# ## Load pre-processed data to check network performance

# dimension of the patch
p1 = 10
p2 = 10
p3 = 10
# weight number
x = 3

# we load a test case
# choose the reasonable region of test patch


# nrrdData = nrrd.read('Case033.nrrd')
# im = nrrdData[0]
# pick = im[285:333, 240:300, 50:]

nrrdData = nrrd.read('Case077.nrrd')
im = nrrdData[0]
pick = im[227:340, 206:246, 30:]


# nrrdData = nrrd.read('Case064.nrrd')
# im = nrrdData[0]
# pick = im[252:412, 145:282, 65:]

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
    for xi in range(x0, xe):
        for yi in range(y0, ye):
            for zi in range(z0,ze):

                if pick[xi,yi,zi] == 0:
                    tips.append([xi+227,yi+206,zi+30,1])
        bar.update()
    return tips



# find the tips for patches with size p
res = findtips(1, p1, p2, p3)
res = np.array(res)
print(res.shape)

f = open('77-black.save', 'wb')
np.save(f, res)
f.close
