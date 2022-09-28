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

#chrom='chr21'
#gene='SOD1'
# gene='LINC00308'
chrom=sys.argv[1]
gene=sys.argv[2]

genefile="genes/"+gene+".pkl"

pkl_file=open(genefile,'rb')
dataset_X,dataset_Y=pickle.load(pkl_file)
dataset_X.shape

# dup_rate=round(1-1.0*pd.DataFrame(dataset_X).drop_duplicates().shape[0]/dataset_X.shape[0],3)

N_snps=dataset_X.shape[1]


if N_snps > 20:
    encoding_dim = 8
elif N_snps <= 20 and N_snps > 4:
    encoding_dim = 4
else:
    encoding_dim = 1


#umap
n_comp=encoding_dim
pca = PCA(n_components=n_comp)
embedding = pca.fit_transform(dataset_X)
embedding.shape


print(pca.explained_variance_ratio_)
sum(pca.explained_variance_ratio_)


# -------- markdown --------
# ## save result

#save top embedding

os.system("mkdir -p pca")

encoded_file="pca/"+gene+".pkl"
with open(encoded_file,'wb') as fw:
    pickle.dump(embedding,fw)



# pkl_file=open(encoded_file,'rb')
# dataset_X=pickle.load(pkl_file)

