#!/usr/bin/env python
import pickle
import json
import csv
import numpy as np
import os
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import argparse
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier


#set randome seed
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) #fix hash seed
	np.random.seed(seed)
	#torch.manual_seed(seed)
	#torch.cuda.manual_seed(seed)
	#torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	#torch.backends.cudnn.benchmark = False
	#torch.backends.cudnn.deterministic = True

seed = 1521024
seed_torch(seed)

#top_k=int(sys.argv[1]) #top k
method=sys.argv[1]
feature_file=sys.argv[2]
ratio=float(sys.argv[3]) #for downsampling train sampels
prefix=sys.argv[4]

top_k=10405 #75584 ,#PC <min(#sample)

# read once a time
dataset_X=pickle.load(open(feature_file,'rb'))
#dataset_X,_=pickle.load(open('../sigSNPs_autoencoder.features.pkl','rb'))
dataset_X.shape

dataset_X=np.array(dataset_X) # extract selected features

_,dataset_Y=pickle.load(open('../chr1/genes/A3GALT2.pkl','rb'))

dataset_X.shape
dataset_Y.shape



# train dataset
train_idx = [int(line.strip()) for line in open("../train_val.balanced.idx", 'r')]
# train_idx = [int(line.strip()) for line in open("../train_val.unique.idx", 'r')]
print(len(train_idx))

# test dataset
te_idx = [int(line.strip()) for line in open("../test.idx", 'r')]

#subsampling

#random.seed(123)
random.shuffle(train_idx)
random.shuffle(te_idx)

train_idx = random.sample(train_idx,int(len(train_idx)*ratio))
random.shuffle(train_idx)

##
x_train=[]
y_train=[]
x_test=[]
y_test=[]

x_train=dataset_X[train_idx]
x_test=dataset_X[te_idx]

y_train=dataset_Y[train_idx]
y_test =dataset_Y[te_idx]

print('x_train:{}'.format(x_train.shape))
print('x_test:{}'.format(x_test.shape))
y_train.shape
y_test.shape

y_train = np.argmax(y_train, axis=1)
y_test_num = np.argmax(y_test, axis=1)

if method=='lr':
    #### LR
    print("\nrunning LR...\n")
    logisticRegr = LogisticRegression(random_state=1991,solver='saga')
    logisticRegr.fit(x_train,y_train)
    y_pred = logisticRegr.predict(x_test)
    y_test_num = np.argmax(y_test, axis=1)
    tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()
    acc = round((tp + tn) * 1. / (tp + fp + tn + fn),3)
    ps = round(tp*1./(tp+fp),3)
    rc = round(tp*1./(tp+fn),3)
    f1=round(2*(ps*rc)/(ps+rc),3)

    print('Accuracy:',(tp+tn)*1./(tp+tn+fp+fn))
    print("Pression: ", ps)
    print("Recall:", rc)
    print("F1: ",2*(ps*rc)/(ps+rc))

    print("TP={}, TN={}, FP={}, FN={}".format(tp,tn,fp,fn))
    with open(prefix+'.out.csv','a') as fw:
        fw.write(','.join([prefix]+list(map(str,[ps,rc,f1,acc])))+'\n')

elif method =='rf':
    ###### predict based top promoters selected by randomforest
    print("\nrunning RandomForest...\n")
    # parameters are from the previous paper
    #clf=RandomForestClassifier(random_state=1991)
    clf=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1991)

    # Train the model using the training sets
    clf.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = clf.predict(x_test)

    y_test_num = np.argmax(y_test, axis=1)
    tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()

    acc = round((tp + tn) * 1. / (tp + fp + tn + fn),3)
    ps = round(tp*1./(tp+fp),3)
    rc = round(tp*1./(tp+fn),3)
    f1=round(2*(ps*rc)/(ps+rc),3)

    print('Accuracy:',(tp+tn)*1./(tp+tn+fp+fn))
    print("Pression: ", ps)
    print("Recall:", rc)
    print("F1: ",2*(ps*rc)/(ps+rc))

    with open(prefix+'.out.csv','a') as fw:
        fw.write(','.join([prefix]+list(map(str,[ps,rc,f1,acc])))+'\n')

elif method=='svm':
    ######## SVM
    print("\nrunning SVM...\n")
    lsvm = svm.SVC(random_state=1991)
    lsvm.fit(x_train,y_train)
    y_pred = lsvm.predict(x_test)
    y_test_num = np.argmax(y_test, axis=1)
    tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()
    acc = round((tp + tn) * 1. / (tp + fp + tn + fn),3)
    ps = round(tp*1./(tp+fp),3)
    rc = round(tp*1./(tp+fn),3)
    f1=round(2*(ps*rc)/(ps+rc),3)

    print('Accuracy:',(tp+tn)*1./(tp+tn+fp+fn))
    print("Pression: ", ps)
    print("Recall:", rc)
    print("F1: ",2*(ps*rc)/(ps+rc))
    print("TP={}, TN={}, FP={}, FN={}".format(tp,tn,fp,fn))
    with open(prefix+'.out.csv','a') as fw:
        fw.write(','.join([prefix]+list(map(str,[ps,rc,f1,acc])))+'\n')

elif method=='adab':
    ######## adab
    print("\nrunning AdaBoostClassifier...\n")
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=1000,random_state=1991)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    y_test_num = np.argmax(y_test, axis=1)
    tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()
    acc = round((tp + tn) * 1. / (tp + fp + tn + fn),3)
    ps = round(tp*1./(tp+fp),3)
    rc = round(tp*1./(tp+fn),3)
    f1=round(2*(ps*rc)/(ps+rc),3)

    print('Accuracy:',(tp+tn)*1./(tp+tn+fp+fn))
    print("Pression: ", ps)
    print("Recall:", rc)
    print("F1: ",2*(ps*rc)/(ps+rc))
    print("TP={}, TN={}, FP={}, FN={}".format(tp,tn,fp,fn))
    with open(prefix+'.out.csv','a') as fw:
        fw.write(','.join([prefix]+list(map(str,[ps,rc,f1,acc])))+'\n')

