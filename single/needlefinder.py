
# coding: utf-8

# # Find TIPS
# ## After training the network, load the weights and apply network to classify each voxel of a test case

from __future__ import division

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
from keras.models import model_from_json
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-n", help="case_num", default=1, type=int)
args = parser.parse_args()

Case = args.n

# weight number
x = 0


# we load the model with the trained weights
model = model_from_json(open('my_model_architecture%d.json'%x).read())
model.load_weights('my_model_weights_2d_%d.h5'%x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# we load a test case
# choose the reasonable region of test patch
nrrdData = nrrd.read(str(Case)+'.nrrd')
im = nrrdData[0]
r, s, t = im.shape
pick = im[r//3:r//3*2, s//3:s//3*2, :]
print(pick.shape)

# Find the tip in the image by computing testing patches at every voxel position
def findtips():
    N = 1
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
    for xi in range(x0, xe-10):
        for yi in range(y0, ye-10):
            vols = [pick[xi:xi+10,yi:yi+10,zi:zi+10] for zi in range(z0,ze-10)]
            # we normalize the data (centered on mean 0 and rescaled in function of the STD)
            volnorm = [ x-np.mean(x) for x in vols]
            volnorm2 = [x/np.std(x) for x in volnorm]
            cube = np.array(volnorm2)
            pro = model.predict_proba(cube, batch_size=32, verbose=False)
            indices = np.where(pro[:,0] > 0.5)
            # we add the coordinates of the center voxel of the patches that tested positive
            for i in indices[0]:
                tips.append([xi+5+r//3,yi+5+s//3,z0+5+i, pro[i,0]])
        bar.update()
    return tips

# find the tips for patches with size p
res = findtips()
rest = np.array(res)
print(rest.shape)
while len(rest) > 1:
    flag = 0
    ini = [0]
    cluster = []
    while flag != 1:
        flag = 1
        if cluster == []:
            cluster = rest[[ini]]
        else :
            cluster = np.concatenate((cluster, rest[ini]), axis=0)
        rest = np.delete(rest,ini,0)
        ini = []
        for ele in cluster:
            for i,j in enumerate(rest):
                if j[2] >= ele[2]:
                    dist = abs(ele[0]-j[0])+abs(ele[1]-j[1])+abs(ele[2]-j[2])
                    if dist < 3:
                        if i not in ini:
                            ini.append(i)
                        flag = 0
    if len(cluster) > 40:
        mid = np.average(cluster[:,:3], axis=0, weights=cluster[:,3])
        final = cluster[:,:3]
print('needle body:', mid)


def pick(coord):
    if int(coord[2]) >= t-2:
        return 0
    for x in range(5):
        for y in range(5):
            if im[int(coord[0])+x-2,int(coord[1])+y-2,int(coord[2])] < 60:
                return 1
    return 0

coord = mid
flag = 1
while flag == 1:
    flag = 0
    temp_coord = [int(coord[0]),int(coord[1]),int(coord[2])]
    temp = im[int(coord[0]),int(coord[1]),int(coord[2])]
    while temp != 0:
        for x in range(5):
            for y in range(5):
                if im[int(coord[0])+x-2,int(coord[1])+y-2,int(coord[2])] < temp:
                    temp = im[int(coord[0])+x-2,int(coord[1])+y-2,int(coord[2])]
                    temp_coord = [int(coord[0])+x-2,int(coord[1])+y-2,int(coord[2])]
        if int(coord[0]) == temp_coord[0] and int(coord[1]) == temp_coord[1]:
            coord = temp_coord
            break
        coord = temp_coord
    # up layer
    if pick([int(coord[0]),int(coord[1]),int(coord[2]+1)]):
        flag = 1
        coord[2] += 1
if temp < 60:
    mid = temp_coord

print('needle tip: ', mid)
mask = np.zeros(im.shape)
for coord in final:
    mask[int(coord[0]),int(coord[1]),int(coord[2])]=1.0
nrrd.write('mask%d.nrrd'%Case, mask, nrrdData[1])

f = open(str(Case)+'-tip.save', 'wb')
np.save(f, [mid])
f.close