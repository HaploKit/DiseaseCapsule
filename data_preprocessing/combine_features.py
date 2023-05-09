#!/usr/bin/env python

import os
import pickle
import numpy as np
import pandas as pd

# only use train dataset for feature selection !!
# train_idx = [int(line.strip())for line in open("../train_val.unique.idx", 'r')]

# test dataset
# te_idx = [int(line.strip()) for line in open("./test.idx", 'r')]


_,Y=pickle.load(open('./chr1/genes/A3GALT2.pkl','rb'))
# Y = Y[train_idx]
y=np.argmax(Y,axis=1)
Y.shape
y.shape

dataset_X = []
header_list = []
flag=0
for gene_file in os.popen('ls chr*/pca/*.pkl').read().strip().split():
    flag+=1
#     if flag%5000==0:print(flag)
#     if flag>200: break
    # chr10/encoder/ABI1.pkl
    gene = gene_file.replace('.pkl', '').replace('/pca/', ':')
    pkl_file = open(gene_file, 'rb')
    gene_X = pickle.load(pkl_file)
#     gene_X = gene_X[train_idx]
    dataset_X.append(gene_X)
    header_list.extend([gene+':'+str(j) for j in range(gene_X.shape[1])])



dataset_X=pd.DataFrame(np.concatenate(dataset_X, axis=1),columns=header_list)
dataset_X.shape

N = dataset_X.shape[1]


fw=open('sigSNPs_pca.features.pkl','wb')
pickle.dump((dataset_X,y),fw)





