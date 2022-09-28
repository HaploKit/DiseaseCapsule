#!/usr/bin/env python


import csv
import numpy as np
import os
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns



# work dir: dolores, /project/als/analysis/qc/rmbatch/final_v2/explainCaps/view_capsules.ipynb

res_rand_kgenes=pd.read_csv('res_rand_kgenes.csv',sep='\t')
res_rand_kgenes[1000:1003]

plt.figure(figsize=[5*2,3.5*2]) #width around 180mm := 7 inches
#font default: ['sans-serif']
ylabels=['Precision','Recall','F1-score','Accuracy']
for i in range(4):
    plt.subplot(2,2,i+1)
    ax=sns.boxplot(x='numOfGenes',y=res_rand_kgenes.columns[i+1],data=res_rand_kgenes)
#     ax = sns.swarmplot(x='numOfGenes',y=res_rand_kgenes.columns[i+1],data=res_rand_kgenes, color=".25")
    if i <2:
        plt.xlabel('')
        ax.set_xticklabels([])
        
    else:
        plt.xlabel('Number of Genes',size=12)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.ylabel(ylabels[i],size=12)
    plt.title(['a','b','c','d'][i],size=14,loc='left',fontweight='bold')
plt.savefig('res_rand_kgenes.pdf',bbox_inches='tight')
print('done');

