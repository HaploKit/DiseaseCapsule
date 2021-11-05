#!/usr/bin/env python

# -------- markdown --------
# # import packages

import json
import csv
import numpy as np
import os
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import pickle
import copy

import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.keras.utils import plot_model

## Setting GPU
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 2 , 'CPU': 48} )
config.gpu_options.allow_growth=True   # assign GPU memory as necessary
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use GPU 1
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

from tensorflow.keras.utils import multi_gpu_model
#GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


from tensorflow.keras import regularizers

from tensorflow.keras.models import Sequential,Model
# from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.utils import get_custom_objects

from tensorflow.keras.layers import Input, Conv1D, Convolution1D,Convolution2D,Reshape,AveragePooling2D,\
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation,AveragePooling1D, GlobalAveragePooling1D,\
    GlobalMaxPooling2D, Flatten, MaxPool1D, Conv2D,MaxPool2D,SeparableConv2D,Conv3D,Add,Dropout,\
    ZeroPadding2D,SeparableConv1D,AveragePooling2D

from tensorflow.keras.constraints import max_norm,unit_norm,min_max_norm

# from keras.utils.vis_utils import plot_model
#from keras.applications.mobilenet import DepthwiseConv2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD,RMSprop,Adam, Nadam, Adagrad
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from keras.backend import sigmoid
from tensorflow.keras.layers import concatenate,ReLU,ThresholdedReLU,LeakyReLU, PReLU,ELU
# from keras.layers.advanced_activations import LeakyReLU, PReLU
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

import argparse
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import keras_metrics as km

from livelossplot.keras import PlotLossesCallback
# keras.backend.image_data_format()
# tf.keras.backend.set_image_data_format('channels_last')
import torch

#set randome seed
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) #fix hash seed
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed = 1521024
seed_torch(seed)

# # Naive CNN

# -------- markdown --------
# ## read data
feature_file,ratio,prefix=sys.argv[1:]
ratio=float(ratio)

model_type='cnn'
# top_k=int(sys.argv[1])
# model_type=sys.argv[2]

if model_type=='cnn':
    top_k=75076

# read once a time

if prefix.split('_')[1]=='allpca':
    dataset_X=pickle.load(open(feature_file,'rb'))
else:
    dataset_X,_=pickle.load(open(feature_file,'rb'))


if dataset_X.shape[1] <top_k:
   top_k=102*102


dataset_X=np.array(dataset_X) # extract selected features
dataset_X=dataset_X[:,:top_k]

_,dataset_Y=pickle.load(open('../chr1/genes/A3GALT2.pkl','rb'))

print(dataset_X.shape)
dataset_Y.shape
top_k=dataset_X.shape[1]

# train dataset
train_idx = [int(line.strip()) for line in open("./train_val.balanced.idx", 'r')]
# train_idx = [int(line.strip()) for line in open("../train_val.unique.idx", 'r')]

# test dataset
te_idx = [int(line.strip()) for line in open("./test.idx", 'r')]


#subsampling

random.seed(123)
random.shuffle(train_idx)
random.shuffle(te_idx)

train_idx = random.sample(train_idx,int(len(train_idx)*ratio))

x_train=[]
y_train=[]
x_test=[]
y_test=[]

x_train=dataset_X[train_idx]
x_test=dataset_X[te_idx]

if model_type=='cnn':
    x_train=x_train.reshape(x_train.shape[0],int(np.sqrt(top_k)),int(np.sqrt(top_k)),1)
    x_test=x_test.reshape(x_test.shape[0],int(np.sqrt(top_k)),int(np.sqrt(top_k)),1)
#     x_train=x_train.reshape(x_train.shape[0],1,int(np.sqrt(top_k)),int(np.sqrt(top_k)))
#     x_test=x_test.reshape(x_test.shape[0],1,int(np.sqrt(top_k)),int(np.sqrt(top_k)))
else:
    x_train=x_train.reshape(x_train.shape[0],top_k,)
    x_test=x_test.reshape(x_test.shape[0],top_k,)

y_train=dataset_Y[train_idx]
y_test =dataset_Y[te_idx]

x_train.shape
x_test.shape
y_train.shape
y_test.shape

# -------- markdown --------
# ## define architecture

maxnorm=3

conv_kwargs2 = {
#     'padding': 'same',
    'strides': 2,
#     'data_format': 'channels_first',
#     'kernel_constraint': max_norm(maxnorm),
#     'kernel_regularizer': regularizers.l2(0.001),
#     'activity_regularizer': regularizers.l1(0.01)
}

dense_kwargs = {
#     'kernel_constraint': max_norm(maxnorm),
#     'kernel_regularizer': regularizers.l2(0.01),
#     'activity_regularizer': regularizers.l1(0.01)
}

bn_kwargs={
    'epsilon':1e-5,
#     'axis':1 # use 1 if channels_first,otherwise -1
}

def als_cnn(lr, optimizer, dense_layer_sizes, dropout_rate, act):
    x = Input(shape=(int(np.sqrt(top_k)), int(np.sqrt(top_k)),1))
    conv=x

    # add traditional conv

    conv = Conv2D(32, (3,3), **conv_kwargs2)(conv)
    conv = Conv2D(32, (3,3), **conv_kwargs2)(conv)
    conv = BatchNormalization(**bn_kwargs)(conv)
    conv = Activation(act)(conv)
#     conv = MaxPool2D(pool_size=(2,2),**mp_kwargs)(conv)

    conv = Conv2D(64, (3, 3),**conv_kwargs2)(conv)
    conv = Conv2D(64, (3, 3),**conv_kwargs2)(conv)
    conv = BatchNormalization(**bn_kwargs)(conv)
    conv = Activation(act)(conv)
#     conv = MaxPool2D(pool_size=(2,2))(conv) # do not use because of worse performance


    flatten = Flatten()(conv)

    d = Dense(128, **dense_kwargs)(flatten)
    d = BatchNormalization()(d)
    d = Activation(act)(d)
    d = Dropout(rate=dropout_rate)(d)

    d = Dense(64, **dense_kwargs)(d)
    d = BatchNormalization()(d)
    d = Activation(act)(d)
    d = Dropout(rate=dropout_rate)(d)

    pred = Dense(2, activation='softmax')(d)
    # Compile model
    model = Model(inputs=[x], outputs=[pred])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer(),
                  metrics=['accuracy']
                  )

    return model

# -------- markdown --------
#  ## trainning & test

param_cnn = {
#     "lr": 0.01,
    "lr": 0.00005,
    "dense_layer_sizes": [0],
#     "batch_size": 128,
#     "epochs": 40,
    "dropout_rate": 0.5,
#     "optimizer": Nadam,
    "optimizer": Adam,
#     "act": "sigmoid","hard_sigmoid"
#     "act": ReLU,ThresholdedReLU,LeakyReLU, PReLU,ELU
    "act": "relu"
}

# callbacks
batch_size=128
epochs=30

def step_decay_schedule(initial_lr=1e-2,min_lr=1e-7, decay_factor=0.75, step_size=4):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return np.max([initial_lr * (decay_factor ** np.floor(epoch/step_size)),min_lr])

    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule(initial_lr=param_cnn['lr'], decay_factor=0.7, step_size=4)

# initial_learning_rate=param_cnn['lr']
# scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=100000,
#     decay_rate=0.8,
#     staircase=True)
# lr_sched = tf.keras.callbacks.LearningRateScheduler(scheduler)

# train the network
print("[INFO] training network...")

model = als_cnn(**param_cnn)

H = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=batch_size,
    epochs=epochs,
#     callbacks=[PlotLossesCallback(plot_extrema=False)],
    callbacks=[lr_sched,PlotLossesCallback(plot_extrema=False)],
#     workers=2,
#     use_multiprocessing=True,
#     shuffle=True,
    verbose=1)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

acc = round((tp + tn) * 1. / (tp + fp + tn + fn),3)
ps = round(tp*1./(tp+fp),3)
rc = round(tp*1./(tp+fn),3)
f1=round(2*(ps*rc)/(ps+rc),3)

print('Accuracy:',(tp+tn)*1./(tp+tn+fp+fn))
print("Pression: ", ps)
print("Recall:", rc)
print("F1: ",2*(ps*rc)/(ps+rc))

# get the mean metrics of last 5 epoches
# acc=round(np.mean(H.history['val_accuracy'][-5:]),3)
# precision=round(np.mean(H.history['val_precision'][-5:]),3)
# recall=round(np.mean(H.history['val_recall'][-5:]),3)
# f1=round(np.mean(H.history['val_f1_score'][-5:]),3)


# print("TP={}, TN={}, FP={}, FN={}".format(tp, tn, fp, fn))

# print('precision:{}, recall:{}, f1:{}, accuracy:{}'.format(
#     precision, recall, f1, acc))


#plt.plot(H.history['accuracy'])
#plt.plot(H.history['val_accuracy'])
#np.max(H.history['accuracy'])
#np.max(H.history['val_accuracy'])



#model.summary()

#plot_model(model,dpi=300,show_shapes=True,to_file='DNN_result/{}.png'.format(model_type))

#save results
with open(prefix+'.out.csv','a') as fw:
        fw.write(','.join([prefix]+list(map(str,[ps,rc,f1,acc])))+'\n')

#with open('DNN_result/cnn.all_chr.top'+str(top_k)+'.out.csv','w') as fw:
#        fw.write(','.join(['mlp']+list(map(str,[ps,rc,f1,acc])))+'\n')
#model.save('DNN_result/cnn.all_chr.top'+str(top_k)+'.h5')
#with open('DNN_result/cnn.all_chr.top'+str(top_k)+'.history.pkl','wb') as fw:
#    pickle.dump(H.history,fw)



