
# coding: utf-8

## Prepare data for training and validation
import glob
import os
import numpy as np
import nrrd
import six.moves.cPickle as pickle
USERPATH = os.path.expanduser("~")
from keras.preprocessing.image import ImageDataGenerator

# path of tips and notips
notipsPath = "/Projects/notips_10-10-10_1.00-1.00-1.00/"
tipsPath = "/Projects/tips_10-10-10_1.00-1.00-1.00/"

# 1 for training, 2 for validation
switch = 1

def loadAllDataFromPath(path, num_extract=200):
    # path in directorty
    output = []
    for file in os.listdir(USERPATH + path):
        if file[0] == '.' :
            continue
        if switch == 1:
            if file == '064' or file == '033' or file == '077':
                continue
        else:
            if file != '064' and file != '033' and file != '077':
                continue
        print("open file:" + USERPATH + path + file)
        cubeTipsPath = glob.glob(USERPATH + path + file + "/*.nrrd")
        # number of samples
        N = len(cubeTipsPath)
        print('number of sample %d' %N)
        cubeTips = []
        for path_i in cubeTipsPath:
            cubeTips.append(nrrd.read(path_i))
        data = [[] for i in range(N)]
        for i in range(N):
            if i < num_extract:
                c = np.array(cubeTips[i][0])  # for patches of size 10,10,20
                if c.shape == (10,10,10):
                    output.append(c)
    output = np.array(output)
    return output

## Create a new dataset
tips = loadAllDataFromPath(tipsPath, 100)
notips = loadAllDataFromPath(notipsPath, 200)

## flatten or data augmentation if needed
# data = [[] for i in range(len(tips))]
# for i in range(len(tips)):
#     data[i] = [tips[i].reshape(50,40)]
# tips = np.array(data)

# datagen = ImageDataGenerator(
#         rotation_range=0.2,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
# y = []
# for k in range(len(tips)):
#     print('batch:' ,k)
#     i = 0
#     x = tips[k]
#     x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 1, 50, 40)
#     for X_batch in datagen.flow(x, batch_size=32):
#         i += 1
#         y.append(X_batch[0])
#         if i == 5:
#             break
# tips = np.array(y)

# data = [[] for i in range(len(notips))]
# for i in range(len(notips)):
#     data[i] = [notips[i].reshape(50,40)]
# notips = np.array(data)

print('tips shape and notips shape:')
print(tips.shape, notips.shape)

target_0 = [0 for i in range(len(notips))]
target_1 = [1 for i in range(len(tips))]
Y_train = np.array(target_0 + target_1)
print('target shape:', Y_train.shape)
X_train = np.array(list(notips)+list(tips))
print('data shape:', X_train.shape)

## **data**: 2D array of randomCubes then tips and **target** 2D array of 0, then 1
# save data for next step
f_Xtrain = open('data_n%d.save'%switch, 'wb')
f_Ytrain = open('label_n%d.save'%switch, 'wb')

pickle.dump(X_train, f_Xtrain, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(Y_train, f_Ytrain, protocol=pickle.HIGHEST_PROTOCOL)

f_Xtrain.close()
f_Ytrain.close()
