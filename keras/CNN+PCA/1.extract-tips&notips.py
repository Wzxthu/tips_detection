
# coding: utf-8

## Extract needle tips and random patchs
import glob
import os
import numpy as np
import nrrd
USERPATH = os.path.expanduser("~")
print(USERPATH)

def getPaths(spacing):
    labelPath = USERPATH + '/DATA/LabelMaps_%.2f-%.2f-%.2f/*' % tuple(spacing)
    print(labelPath)
    casePath = glob.glob(labelPath)
    volumePath = [cp + '/case.nrrd' for cp in casePath]
    return volumePath, casePath

def getVolumeAndNeedle(k, casePath):
    '''
    Outputs the volume and needles paths
    '''
    needlePath  = glob.glob(casePath[k] + '/needle*.nrrd')
    return needlePath

def getTipsPos(ndls):
    '''
    Read the needle labelmaps and outputs the tip positions
    '''
    needleTips = []
    for n in ndls:
        ndl = nrrd.read(n)[0]
        ztest = (np.where(ndl>0)[2])
        if len(ztest):
            zmax = np.where(ndl>0)[2].max()
            xl,yl = np.where(ndl[...,zmax]>0)
            xmax = int(np.mean(xl))
            ymax = int(np.mean(yl))
            needleTips.append([xmax, ymax, zmax])
    return needleTips

def getTipsPath(casePath, spacing):
    '''
    Returns path where to save the tips
    '''
    return USERPATH + '/Projects/tips_%d-%d-%d_%.2f-%.2f-%.2f/%s' %(tuple(patchsize)+tuple(spacing)+(casePath.split('/')[-1],))

def getNoTipsPath(casePath, spacing):
    '''
    Return path where to save the random patchs
    '''
    return USERPATH + '/Projects/notips_%d-%d-%d_%.2f-%.2f-%.2f/%s' %(tuple(patchsize)+tuple(spacing)+(casePath.split('/')[-1],))

def saveTips(tipsPos, casePath, spacing, patchsize):
    '''
    Save the tips
    '''
    print('open: ', casePath)
    tipPath = getTipsPath(casePath, spacing)
    vol = nrrd.read(casePath + '/case.nrrd')[0]
    for i, tipPos in enumerate(tipsPos):
        x, y, z = tipPos
        xmin, ymin, zmin = np.array(patchsize)//2
        xmin = xmin//spacing[0]
        ymin = ymin//spacing[1]
        zmin = zmin//spacing[2]
        tip = vol[x-xmin:x+xmin, y-ymin:y+ymin, z-zmin:z+zmin]
        createDir(tipPath)
        nrrd.write(tipPath + '/tip-%d.nrrd'%i, tip)

def saveNoTips(tipsPos, numberOfSamples, casePath, spacing, patchsize):
    '''
    Pick suitable region
    '''
    print('open: ', casePath)
    vol = nrrd.read(casePath + '/case.nrrd')[0]
    r1, s1, t1 = vol.shape
    # pick approximate region
    pick = vol[r1//3:r1//3*2, s1//3:s1//3*2+20, t1//2+15:t1//3*2+20]
    r2, s2, t2 = pick.shape
    print('vol shape:', vol.shape)
    print('pick shape:', pick.shape)
    '''
    Find voiding region of tips
    '''
    region = []
    ban = 9
    for tip in tipsPos:
        x, y, z = tip
        for i in range(ban):
            for j in range(ban):
                region.append([x-r1//3+i-ban//2, y-s1//3+j-ban//2])
    print('region:', np.shape(region))
    '''
    Save the random cubes
    '''
    notipPath = getNoTipsPath(casePath, spacing)
    for i in range(numberOfSamples):
        xmin, ymin, zmin = np.array(patchsize)//2
        xmin = xmin//spacing[0]
        ymin = ymin//spacing[1]
        zmin = zmin//spacing[2]
        x = np.random.randint(xmin,r2-xmin)
        y = np.random.randint(ymin,s2-ymin)
        z = np.random.randint(zmin,t2-zmin)
        # exclude region where needles locate
        while [x, y] in region:
            x = np.random.randint(xmin,r2-xmin)
            y = np.random.randint(ymin,s2-ymin)
            print('find contradiction!')
        notip = pick[x-xmin:x+xmin, y-ymin:y+ymin, z-zmin:z+zmin]
        createDir(notipPath)
        nrrd.write(notipPath + '/notip-%d.nrrd'%i, notip)

def createDir(directory):
    '''
    Create a directory if it doesn't exist. Do nothing otherwise.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
    return 0

def extractTips(spacing, patchsize):
    '''
    Extract all the tips from all the labelmaps
    '''
    _, casePath = getPaths(spacing)
    for k, path in enumerate(casePath):
        ndls = getVolumeAndNeedle(k, casePath)
        tipsPos = getTipsPos(ndls)
        saveTips(tipsPos, casePath[k], spacing, patchsize)

def extractNoTips(spacing, patchsize, numberOfSamples):
    '''
    Extract random cases
    '''
    _, casePath = getPaths(spacing)
    for k, path in enumerate(casePath):
        ndls = getVolumeAndNeedle(k, casePath)
        tipsPos = getTipsPos(ndls)
        saveNoTips(tipsPos, numberOfSamples, casePath[k], spacing, patchsize)

'''
Extract tips&notips according to different spacing&patchsize
'''
print('begin extracting:')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
for spacing in [[1,1,1]]:
    for patchsize in [[10,10,10], [10,10,20]]:
        print('spacing:', spacing)
        print('patchsize:', patchsize)
        print('extract tips:')
        extractTips(spacing, patchsize)
        print('extract notips:')
        extractNoTips(spacing, patchsize, 200)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('All done!')

