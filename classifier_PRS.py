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


top_k='_p0.05'
vcf='/export/scratch3/vincent/project/als/GWAS2019_NL_QC/qc/rm_batch_snp/vcf/vcf_p0.05/all_chrs.vcf'

#top_k='_p5e-8'
#vcf='/export/scratch3/vincent/project/als/GWAS2019_NL_QC/qc/rm_batch_snp/vcf/vcf_p5e-8/all_chrs.vcf'

method=sys.argv[1]


vcf_df=pd.read_csv(vcf,sep='\t',index_col=2)

labels_file='/export/scratch3/vincent/project/als/analysis/qc/rmbatch/labels.csv'
labels_df = pd.read_csv(labels_file, index_col=0)
sample_ids = labels_df.FID.tolist()
sample_ids[:4]

# the output is a vector for probability in DL
lab_num = {1: [1, 0], # negative, max_idx=0
           2: [0, 1]} # positive, max_idx=1
pheno_new = []
for i in labels_df.Pheno.tolist():
    pheno_new.append(lab_num[i])

vcf_df = vcf_df.iloc[:,8:]
vcf_df.columns = [a.split('_')[0] for a in vcf_df.columns]


vcf_df=vcf_df[sample_ids]

# mapping dictionary
genotype2num = {"0/0":'0',
                "0/1":'1',
                "1/0":'1',
                "1/1":'2',
                "./.":'-1'}

vcf_df.replace(genotype2num,inplace=True)
vcf_df.shape

vcf_df[:3]

dataset_X=np.array(vcf_df).astype('float32').T
# N = dataset_X.shape[1]

dataset_Y = np.array(pheno_new)
dataset_Y = dataset_Y.reshape((len(dataset_Y), dataset_Y.shape[1]))


dataset_X.shape
dataset_Y.shape
dataset_X[:3]
dataset_Y[:3]

for _ in range(9):
	for ratio in [0.1,0.3,0.5,0.7,0.9,1.0]:
	    # ratio=float(sys.argv[2]) #for downsampling train sampels
	    prefix='lr_none_'+str(ratio) #PRS based
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
	    random.seed(123)
	    random.shuffle(train_idx)


	    x_train=[]
	    y_train=[]
	    x_test=[]
	    y_test=[]

	    x_train=dataset_X[train_idx]
	    x_test=dataset_X[te_idx]

	    y_train=dataset_Y[train_idx]
	    y_test =dataset_Y[te_idx]

	    x_train.shape
	    x_test.shape
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

	        with open('rf.all_chr.top'+str(top_k)+'.out.csv','w') as fw:
	            fw.write(','.join(['all_chrs']+list(map(str,[ps,rc,f1,acc])))+'\n')

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
	        with open('svm.all_chr.top'+str(top_k)+'.out.csv','w') as fw:
	            fw.write(','.join(['all_chrs']+list(map(str,[ps,rc,f1,acc])))+'\n')

	    elif method=='adab':
	        ######## adab
	        print("\nrunning AdaBoostClassifier...\n")
	        clf = AdaBoostClassifier(random_state=1991)
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
	        with open('adab.all_chr.top'+str(top_k)+'.out.csv','w') as fw:
	            fw.write(','.join(['all_chrs']+list(map(str,[ps,rc,f1,acc])))+'\n')

