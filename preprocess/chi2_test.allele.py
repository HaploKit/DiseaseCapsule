#!/usr/bin/env python

import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

import pandas as pd
import sys
from multiprocessing import Pool

vcf=sys.argv[1]
outfile=sys.argv[2] # p value outfile

df = pd.read_csv(vcf, sep='\t', skiprows=30)
print("old dataframe shape:{}".format(df.shape))
batch_df = pd.read_csv("sample_batch.list", sep="\t", header=None)

#filtering
batches=['c1','c3','c5','c44']
#batches=['c5','c44'] #skip c1,c3 because of small sample size
pvals=[]
for m in range(len(batches)-1):
    for n in range(m+1,len(batches)):

        batch_df2 = batch_df.loc[((batch_df[1]==batches[m]) & (batch_df[2]==1)) | ((batch_df[1]==batches[n]) & (batch_df[2]==1)),:]

        df2 = df[list(df.columns[:9])+list(batch_df2[0])]
        print("new dataframe shape:{}".format(df2.shape))

        def chi2_test(i):
            '''test based on allele counts rather than genotype
            '''
            gt=df2.iloc[i,9:]
            snp_df = pd.DataFrame({'genotype': list(gt), 'batch': list(batch_df2[1])})
            geno_df=pd.crosstab(snp_df.genotype, snp_df.batch)
            allele_dict={}
            for idx in geno_df.index:
                for a in idx.split('/'):
                    if a in allele_dict:
                        allele_dict[a]+=np.array(geno_df.loc[idx])
                    else:
                        allele_dict[a]=np.array(geno_df.loc[idx])
            tb = np.array(list(allele_dict.values()))
            p = chi2_contingency(tb)[1]
            return (df2.iloc[i,2],p)


        pool=Pool(48)
        out=pool.map(chi2_test,range(df2.shape[0]),chunksize=1)
        pool.close()
        pool.join()

        out2=[str(out[i][0])+'\t'+str(out[i][1])+'\t'+batches[m]+'_'+batches[n] for i in range(len(out))]
        pvals.extend(out2)


#p_adj=multipletests(pvals, method='bonferroni')[1]


#chr:pos pval pval_corrected
with open(outfile,'w') as fw:
    fw.write('\n'.join(pvals))


