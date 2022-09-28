#!/usr/bin/env python

# -------- markdown --------
# ## import packages

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import os
import sys
import random
import copy
import json
import pickle
from sklearn.decomposition import PCA
import umap


# mapping dictionary
genotype2num = {"0/0":'0',
                "0/1":'1',
                "1/0":'1',
                "1/1":'2',
                "./.":'-1'}

vcf='/export/scratch3/vincent/project/als/GWAS2019_NL_QC/qc/rm_batch_snp/vcf/vcf_p0.05/all_chrs.vcf'
genotypes=[]
i=0
with open(vcf) as fr:
    for line in fr:
        i+=1
        if i%10000==0:print("processing {} line...".format(i))
        if line.startswith("#"):
            continue
        else:
            a=line.strip().split()
            genotypes.append([genotype2num[g] for g in a[9:]])


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



dataset_X = np.array(genotypes).astype('float32').T
genotypes=[]

#dataset_X = dataset_X.reshape((len(dataset_X), dataset_X.shape[1]))

dataset_Y = np.array(pheno_new)
#dataset_Y = dataset_Y.reshape((len(dataset_Y), dataset_Y.shape[1]))

print(dataset_X.shape)
print(dataset_Y.shape)

N_snps=dataset_X.shape[1]


if N_snps > 20:
    encoding_dim = 75584
elif N_snps <= 20 and N_snps > 4:
    encoding_dim = 4
else:
    encoding_dim = 1


print('running PCA...')

n_comp=encoding_dim

#Number of components to keep. if n_components is not set all components are kept:
#n_components == min(n_samples, n_features)

pca = PCA()
#pca = PCA(n_components=n_comp)
embedding = pca.fit_transform(dataset_X)
embedding.shape


print(pca.explained_variance_ratio_)
sum(pca.explained_variance_ratio_)


# -------- markdown --------
# ## save result

#save top embedding

encoded_file="allGenes_pca.pkl"
with open(encoded_file,'wb') as fw:
    pickle.dump(embedding,fw)



# pkl_file=open(encoded_file,'rb')
# dataset_X=pickle.load(pkl_file)

