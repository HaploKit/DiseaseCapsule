#!/usr/bin/env python

# -------- markdown --------
# ## import packages

import pandas as pd
import numpy as np
from itertools import product
import os
import sys
import random
import copy
import pickle


#chrom='chr21'
chrom=sys.argv[1]

# gene to strand
gene2strand={}
with open('/export/scratch3/vincent/project/als/analysis/qc/rmbatch/v3_topSNPs/gene_strand.txt') as fr:
    for line in fr:
        gene,strand=line.strip().split()
        gene2strand[gene]=strand

# read GWAS p value file
pval_file='/export/scratch3/vincent/project/als/analysis/qc/rmbatch/snp_analysis/v2/plink.assoc.rmbatch'
snp2p={}
with open(pval_file,'r') as fr:
    i=0
    for line in fr:
        if i==0:
            i+=1
            continue
        a=line.split()
        snp2p[a[1]]=float(a[-2])
# snp2p

# read gene to snp file

gene2snp={}
# corrdinates of snps should be sorted
with open(chrom+".variant_function.uniqGene",'r') as fr:
    for line in fr:
        chromosome,snp,gene=line.split()[:3]
        if gene in gene2snp:
            gene2snp[gene].append(chromosome+':'+snp)
        else:
            gene2snp[gene]=[chromosome+':'+snp]
len(gene2snp)

# gene2snp
'''
topk=128 #top k SNPs sorted by GWAS P value

del_genes=[] # delete the gene if the min GWAS P value of snps in this gene >0.05
for gene in gene2snp.keys():
    if np.min([snp2p[snp] for snp in gene2snp[gene]]) > 0.05:
        del_genes.append(gene)
        continue

    if len(gene2snp[gene])>topk:
        tmp_snp2p={}
        for snp in gene2snp[gene]:
            tmp_snp2p[snp]=snp2p[snp]
        snps=sorted(tmp_snp2p.keys(),key=lambda x :tmp_snp2p[x],reverse=False)[:topk]

        if gene2strand[gene] == '+':
            gene2snp[gene]=sorted(snps,key=lambda x :int(x.split(':')[-1]),reverse=False)
        elif gene2strand[gene] == '-':
            gene2snp[gene]=sorted(snps,key=lambda x :int(x.split(':')[-1]),reverse=True)
        else:
            raise Exception("Cannot find gene:{} in the gene annotation file!".format(gene))


for gene in del_genes:
    del gene2snp[gene]
len(gene2snp)
'''



vcf='/export/scratch3/vincent/project/als/GWAS2019_NL_QC/qc/rm_batch_snp/vcf/vcf_p0.05/'+chrom+'.vcf'
snp2genotype={}
with open(vcf) as fr:
    for line in fr:
        if line.startswith("##"):
            continue
        elif line.startswith("#CHROM"):
            header=line.strip().split()[9:]
        else:
            a=line.strip().split()
            snp2genotype[a[2]]='\t'.join(a[9:])


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


# mapping dictionary
genotype2num = {"0/0":'0',
                "0/1":'1',
                "1/0":'1',
                "1/1":'2',
                "./.":'-1'}

for gene in gene2snp.keys():
#     gene='LINC00308'
    print('processing gene: {}'.format(gene))

    genotypes = [snp2genotype[snp] for snp in gene2snp[gene]]
    genotypes2=[]
    for genotype in genotypes:
        genotypes2.append([genotype2num[geno] for geno in genotype.split()])
    dataset_X = np.array(genotypes2).astype('float32').T
    dataset_X = dataset_X.reshape((len(dataset_X), dataset_X.shape[1]))

    dataset_Y = np.array(pheno_new)
    dataset_Y = dataset_Y.reshape((len(dataset_Y), dataset_Y.shape[1]))

    print(dataset_X.shape)
    print(dataset_Y.shape)
    os.system("mkdir -p genes")
    outfile="genes/"+gene+".pkl"
    fw = open(outfile,'wb')
    # pickle.dump(gene,'wb')
    pickle.dump((dataset_X,dataset_Y),fw)
    fw.close()



# pkl_file=open(outfile,'rb')
# dataset_X,dataset_Y=pickle.load(pkl_file)

