import numpy as np
import nrrd

mid = np.load('tips+color+promax.save')
print(mid.shape)

nrrdData = nrrd.read('Case064.nrrd')
im = nrrdData[0]
mask = np.zeros(im.shape)
num = 0
for coord in mid:
    if im[int(coord[0]),int(coord[1]),int(coord[2])] < 100:
        num += 1
        mask[int(coord[0]),int(coord[1]),int(coord[2])]=1.0
print(num)
nrrd.write('mask-latest-64-color100+promax.nrrd', mask, nrrdData[1])


