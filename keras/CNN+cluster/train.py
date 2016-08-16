
# coding: utf-8

from __future__ import division
import os
import numpy as np
import nrrd
USERPATH = os.path.expanduser("~")
print(USERPATH)
import six.moves.cPickle as pickle
import random
import multiprocessing
num_cores = multiprocessing.cpu_count()

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils, generic_utils
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline


# Load the dataset
f_Xdata = open('data_n3.save', 'rb')
f_Ydata = open('label_n3.save', 'rb')

X_data = pickle.load(f_Xdata)
print(X_data.shape)
X_data = X_data.astype('float32')

# normalize the raw data
X_data -= np.mean(X_data)
X_data /= np.std(X_data)
Y_data= pickle.load(f_Ydata)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y_data)
Y_data = encoder.transform(Y_data)

label = Y_data
data = X_data

print("Data shape and label shape")
print(data.shape, label.shape)

# init the global var
model = 0

def create_baseline():

    nb_classes = 1

    # create model
    global model
    model = Sequential()

    # model.add(Convolution2D(20, 3, 3, border_mode='valid',
    #                         input_shape=(10,10,10)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(40, 3, 3))
    # model.add(Activation('relu'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))

    # model.add(Convolution2D(40, 5, 3, border_mode='same' ))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(40, 5, 3, border_mode='same' ))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))

    # model.add(Flatten())
    # model.add(Dense(480))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(480))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    # model.add(Dense(nb_classes))
    # model.add(Activation('sigmoid'))

    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(10,10,10)))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))



    model.add(Convolution2D(10, 10, 10, border_mode='same'))
    # model.add(Flatten())
    # model.add(Dense(nb_classes))

    # model.add(Flatten())
    # model.add(Dropout(0.5))
    # model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

seed = 7
# np.random.seed(seed)
estimators = []
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=50,
                                          batch_size=64, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=label, n_folds=4, shuffle=True, random_state=seed)

results = cross_val_score(pipeline, data, label, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

json_string = model.to_json()
x = 0
model.save_weights('my_model_weights_2d_%d.h5'%x, overwrite=True)
open('my_model_architecture%d.json'%x, 'w').write(json_string)


f_Xdata = open('data_n2.save', 'rb')
f_Ydata = open('label_n2.save', 'rb')

# we load the data via pickle
X_data = pickle.load(f_Xdata)
Y_data= pickle.load(f_Ydata)

# we shuffle the data
index = [i for i in range(len(X_data))]
random.shuffle(index)
X_data = X_data[index]
Y_data = Y_data[index]


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y_data)
Y_data = encoder.transform(Y_data)

# we make sure the data is in the right format
X_data = X_data.astype('float32')
X_data  -= np.mean(X_data)
X_data /= np.std(X_data)

X_test = X_data
Y_test = Y_data

# we load the model with the trained weights
model = model_from_json(open('my_model_architecture%d.json'%x).read())
model.load_weights('my_model_weights_2d_%d.h5'%x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# we check what was the performance of this CNN on the trained data
Y_predict = model.predict_classes(X_test[:][:], batch_size=64)
kk = model.predict_proba(X_test[:][:], batch_size=64)
print('predict error:')
print('1 for tip, 0 for notip:')
tip = 0
notip = 0
for i in range(len(Y_predict)):
    # if Y_predict[i][0] == 1:
    #     print(kk[i])
    if Y_test[i] != Y_predict[i][0]:
        if Y_test[i] == 1:
            tip += 1
        else:
            notip += 1
        print(Y_predict[i], Y_test[i])
print('there are ', len(Y_predict), ' samples!')
print('error: ', tip, 'tips and ', notip, 'notips')

