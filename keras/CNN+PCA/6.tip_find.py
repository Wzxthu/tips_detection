
# coding: utf-8

## find tips from middle of needles
import numpy as np
import nrrd

Case_num = 64
threshold = 15

nrrdData = nrrd.read('Case0%d.nrrd'%Case_num)
im = nrrdData[0]

mid = np.load(str(Case_num)+'-mid.save')
print(mid.shape)

## search upper layer
def pick(coord):
    if int(coord[2]) >= 100:
        return 0
    for x in range(5):
        for y in range(5):
            if im[int(coord[0])+x-2,int(coord[1])+y-2,int(coord[2])] < threshold:
                return 1
    return 0

update = []
for coord in mid:
    flag = 1
    while flag == 1:
        flag = 0
        temp_coord = [int(coord[0]),int(coord[1]),int(coord[2])]
        temp = im[int(coord[0]),int(coord[1]),int(coord[2])]
        # search in a circle
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
    if temp < threshold and temp_coord not in update:
        update.append(temp_coord)
mid = update
print(np.shape(mid))
print(mid)

f = open(str(Case_num)+'-tips.save', 'wb')
np.save(f, mid)
f.close