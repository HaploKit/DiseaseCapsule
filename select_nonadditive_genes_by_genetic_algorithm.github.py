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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
from torch.nn.functional import relu,tanh
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from livelossplot import PlotLosses
from torchsummary import summary

from tensorflow.keras.utils import plot_model
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

import geatpy as ea

import warnings
warnings.filterwarnings('ignore')

# -------- markdown --------
# # define models

input_gene_file,iter_k=sys.argv[1:]


class ConvCaps2D(nn.Module):
    def __init__(self):
        super(ConvCaps2D, self).__init__()
        # The paper suggests having 32 8D capsules
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels = primary_capslen,
                                                 kernel_size=(1,ks), stride=stride) for _ in range(filters)])

    def squash(self, tensor, dim=-1):
        norm = (tensor**2).sum(dim=dim, keepdim = True) # norm.size() is (None, 1152, 1)
        scale = norm / (1 + norm) # scale.size()  is (None, 1152, 1)
        return scale*tensor / torch.sqrt(norm)

    def forward(self, x):
        outputs = [capsule(x).view(x.size(0), primary_capslen, -1) for capsule in self.capsules] # 32 list of (None, 1, 8, 36)
        outputs = torch.cat(outputs, dim = 2).permute(0, 2, 1)  # outputs.size() is (None, 1152, 8)
        return self.squash(outputs)


class Caps1D(nn.Module):
    def __init__(self):
        super(Caps1D, self).__init__()
        self.num_iterations = num_iterations
        self.num_caps = 2 # equals to class number
        self.num_routes= (int((neurons-ks)/stride)+1)*filters
        print('num_routes:{}'.format(self.num_routes))
        self.in_channels=primary_capslen
        self.out_channels=digital_capslen

        self.W = nn.Parameter(torch.randn(self.num_caps,self.num_routes, self.in_channels, self.out_channels)) # class,weight,len_capsule,capsule_layer
#         self.W = nn.Parameter(torch.randn(3, 3136, 8, 32)) # num_caps, num_routes, in_channels, out_channels


    def softmax2(self, x, axis=-1):
        ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
        return ex/K.sum(ex, axis=axis, keepdims=True)

    def softmax(self, x, dim = 1):
        transposed_input = x.transpose(dim, len(x.size()) - 1)
#         xxx=transposed_input.contiguous().view(-1, transposed_input.size(-1))
#         print('xxx:{}'.format(xxx.shape))#(2x16xsamples,2336)
#         softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
        # for 2 dimensional data, F.softmax() uses dim=-1 by default,but not in non-two-dimensional data
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)),dim=-1)
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(x.size()) - 1)

    def squash(self, tensor, dim=-1):
        norm = (tensor**2).sum(dim=dim, keepdim = True) # norm.size() is (None, 1152, 1)
        scale = norm / (1 + norm)
        return scale*tensor / torch.sqrt(norm)

    # Routing algorithm
    def forward(self, u):
        # u.size() is (None, 1152, 8)
        '''
        From documentation
        For example, if tensor1 is a j x 1 x n x m Tensor and tensor2 is a k x m x p Tensor,
        out will be an j x k x n x p Tensor.

        We need j = None, 1, n = 1152, k = 10, m = 8, p = 16
        '''

        u_ji = torch.matmul(u[:, None, :, None, :], self.W) # u_ji.size() is (None, 10, 1152, 1, 16)

        b = Variable(torch.zeros(u_ji.size())) # b.size() is (None, 10, 1152, 1, 16)
        b = b.to(device) # using gpu

        for i in range(self.num_iterations):
            c = self.softmax(b, dim=2)
            v = self.squash((c * u_ji).sum(dim=2, keepdim=True)) # v.size() is (None, 10, 1, 1, 16)

            if i != self.num_iterations - 1:
                delta_b = (u_ji * v).sum(dim=-1, keepdim=True)
                b = b + delta_b

        # Now we simply compute the length of the vectors and take the softmax to get probability.
        v = v.squeeze()
        classes = (v ** 2).sum(dim=-1) ** 0.5
#         print('classes:{}'.format(classes.shape))
        classes = F.softmax(classes,dim=-1) # This is not done in the paper, but I've done this to use CrossEntropyLoss.
        return classes


# for name, param in model.named_parameters():
#     if param.device.type != 'cuda:0':
#         print('param {}, not on GPU'.format(name))

class CapsNet(nn.Module):
    def __init__(self):
#         super().__init__() #py3
        super(CapsNet, self).__init__() #py2
        self.fc1 = nn.Linear(top_k,neurons)
        self.dropout1 = nn.Dropout(p=dropout)
#         self.fc2 = nn.Linear(256,64)
#         self.out_channel=16
#         self.conv1 = nn.Conv2d(in_channels = 1, out_channels = self.out_channel, kernel_size = (1,3), stride = 1)
#         self.conv1_bn = nn.BatchNorm2d(self.out_channel)

#         self.conv2 = nn.Conv2d(in_channels = self.out_channel, out_channels = 2*self.out_channel, kernel_size = (3,3), stride = 2)
#         self.conv2_bn = nn.BatchNorm2d(2*self.out_channel)

#         self.conv3 = nn.Conv2d(in_channels = 2*self.out_channel, out_channels = 2*2*self.out_channel, kernel_size = (3,3), stride = 1)
#         self.conv3_bn = nn.BatchNorm2d(2*2*self.out_channel)

        self.primaryCaps = ConvCaps2D()
        self.digitCaps = Caps1D()


    def forward(self, x):
        x = act(self.dropout1(self.fc1(x)))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.conv1_bn(self.conv1(x)))
#         x = F.relu(self.conv2_bn(self.conv2(x)))
#         x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.primaryCaps(x)
        x = self.digitCaps(x)

        return x



# a = Variable(torch.randn(5,2))
# F.softmax(a)
# F.softmax(a,dim=-1)


top_k=75584

#use the best param, id:89
neurons=150
dropout=0.5
primary_capslen=4
digital_capslen=16
ks=5
stride=2
filters=32
num_iterations=3 #danymic routing iterations

##
initial_lr=0.0001
batch_size=128
epochs=30
act=relu

device = torch.device('cuda:1')
model = CapsNet()
model.load_state_dict(torch.load("../../capsule_pca.best_model.pt")) #recommended officically
model.eval()
model.to(device);
# model.fc1.weight.data


# read once a time
dataset_X,_=pickle.load(open('../sigSNPs_pca.features.pkl','rb'))
print(dataset_X.shape)

dataset_X=np.array(dataset_X) # extract selected features

_,dataset_Y=pickle.load(open('../../../chr1/genes/A3GALT2.pkl','rb'))

dataset_X.shape
dataset_Y.shape





# train dataset
# train_idx = [int(line.strip()) for line in open("../train_val.unique.idx", 'r')]
train_idx = [int(line.strip()) for line in open("../../train_val.balanced.downsample_control.idx", 'r')]
x_train=dataset_X[train_idx]

# test dataset
te_idx = [int(line.strip()) for line in open("../../test.idx", 'r')]
x_test=dataset_X[te_idx]
x_train.shape
x_test.shape


# x_train=x_train.reshape(x_train.shape[0],1,1,x_train.shape[1])

# x_test=x_test.reshape(x_test.shape[0],1,1,x_test.shape[1])

y_train =dataset_Y[train_idx]

y_train = np.argmax(y_train, axis=1)

y_test =dataset_Y[te_idx]

y_test = np.argmax(y_test, axis=1)
y_test.shape

# in_train=Variable(torch.tensor(x_train[:500]).to(device))
# y_pred = model(in_train).detach().cpu().numpy()
# y_pred = np.argmax(y_pred, axis=1)
# y_true = y_train[:500]

# -------- markdown --------
# # Compute C_ij for training data (need to split dataset then merge)

def compute_average_cij_for_testdata(x_test,y_test):
    in_test=Variable(torch.tensor(x_test).to(device))
    y_pred = model(in_test).detach().cpu().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    y_true = y_test

    c=np.load('C_ij.npy')
    c=c.reshape(c.shape[0],c.shape[1],32,73)
    c=np.sum(c,axis=-1)

    num_classes=2
    phenotype = ['Healthy','ALS']
    pheno2coefficients={}
    count = {}
    for i in range(len(c)):
        ind = int(y_test[i])     #phenotype
        if ind in pheno2coefficients.keys():
            pheno2coefficients[ind] = pheno2coefficients[ind] + c[i]   #sum of coupling coefficients for each phenotype
            count[ind] = count[ind] + 1   #sample counts for each type
        else:
            pheno2coefficients[ind] = c[i]
            count[ind] = 1

    total = np.zeros((c.shape[1],c.shape[-1]))

    #type average coupling coefficients
    for i in range(num_classes):
        average = pheno2coefficients[i]/count[i]   #sum/count = average
        total[i] = average[i]
#     print(total)
    Cij=total
    return Cij


def compute_average_cij_for_traindata(x_train,y_train):
    k=int(x_train.shape[0]/1000)+1
    y_pred=np.array([])
    for i in range(k):
        upper = min(x_train.shape[0],(i+1)*1000)
    #     print(i,upper)
        in_train=Variable(torch.tensor(x_train[(i*1000):upper]).to(device))
        y_pred0 = model(in_train).detach().cpu().numpy()
        y_pred0 = np.argmax(y_pred0, axis=1)
        y_pred = np.concatenate([y_pred,y_pred0])
        o=os.system('mv C_ij.npy C_ij.{}.npy'.format(i))
    y_true = y_train
    y_pred.shape
    #merge all Cij
    c=np.array([])
    for i in range(k):
        c0=np.load('C_ij.{}.npy'.format(i))
        if i==0:
            c=c0
        else:
            c=np.vstack([c,c0])

    c=c.reshape(c.shape[0],c.shape[1],32,73)
    c=np.sum(c,axis=-1)

    num_classes=2
    phenotype = ['Healthy','ALS']
    pheno2coefficients={}
    count = {}
    for i in range(len(c)):
        ind = int(y_train[i])     #phenotype
        if ind in pheno2coefficients.keys():
            pheno2coefficients[ind] = pheno2coefficients[ind] + c[i]   #sum of coupling coefficients for each phenotype
            count[ind] = count[ind] + 1   #sample counts for each type
        else:
            pheno2coefficients[ind] = c[i]
            count[ind] = 1

    total = np.zeros((c.shape[1],c.shape[-1]))

    #type average coupling coefficients
    for i in range(num_classes):
        average = pheno2coefficients[i]/count[i]   #sum/count = average
        total[i] = average[i]
    Cij=total
    Cij.shape
    return Cij


# compute_average_cij_for_traindata(y_train)

# print("\nrunning LR...\n")
# logisticRegr = LogisticRegression(random_state=1991,solver='saga')
# logisticRegr.fit(x_train,y_train)
# y_pred = logisticRegr.predict(x_test)
# y_test_num = y_test
# # y_test_num = np.argmax(y_test, axis=1)
# tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()
# acc = round((tp + tn) * 1. / (tp + fp + tn + fn),3)
# ps = round(tp*1./(tp+fp),3)
# rc = round(tp*1./(tp+fn),3)
# f1=round(2*(ps*rc)/(ps+rc),3)

# print('Accuracy:',(tp+tn)*1./(tp+tn+fp+fn))
# print("Pression: ", ps)
# print("Recall:", rc)
# print("F1: ",2*(ps*rc)/(ps+rc))

# dump(logisticRegr, 'logisticRegr.joblib')
logisticRegr = load('../logisticRegr.joblib')


X,Y=pickle.load(open('../sigSNPs_pca.features.pkl','rb'))
print(X.shape)
print(Y.shape)
X2=np.array(copy.deepcopy(X))

def compute_diff_acc(gene_matrix):
    #(population_size, gene_set_size)
    all_input_genes=[]
    with open(input_gene_file) as fr:
        for line in fr:
            all_input_genes.append(line.strip())
    diff_accs=[]
    for j in range(gene_matrix.shape[0]):
        gene_vector=gene_matrix[j]
#         print('gene_vector:{}'.format(gene_vector))
        target_genes={}
        target_genes={gene:1 for k,gene in enumerate(all_input_genes) if gene_vector[k]==1}
#         print('target genes number:{}'.format(len(target_genes)))

        #compute acc
        x_train=X.iloc[train_idx] #downsample control such that balanced train case:control
#         x_test=X.iloc[te_idx]
#         x_test=np.array(x_test)
#         y_test =Y[te_idx]
        x_train=np.array(x_train)
        y_train =Y[train_idx]

        for i in range(X.shape[1]):
            if X.columns[i].split(':')[1] not in target_genes: #there are some duplicated genes located on different chrs
                x_train[:,i]=0.

        #LR
        y_pred = logisticRegr.predict(x_train)
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        lr_acc = round((tp + tn) * 1. / (tp + fp + tn + fn),3)

        #Capsnet
        x_train=x_train.reshape(x_train.shape[0],1,1,x_train.shape[1])
#         x_test=x_test.reshape(x_test.shape[0],1,1,x_test.shape[1])

        k=int(x_train.shape[0]/1000)+1
        y_pred=np.array([])
        for i in range(k):
            upper = min(x_train.shape[0],(i+1)*1000)
        #     print(i,upper)
            in_train=Variable(torch.tensor(x_train[(i*1000):upper]).to(device))
            y_pred0 = model(in_train).detach().cpu().numpy()
            y_pred0 = np.argmax(y_pred0, axis=1)
            y_pred = np.concatenate([y_pred,y_pred0])
        y_true = y_train

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = round((tp + tn) * 1. / (tp + fp + tn + fn),3)
        precision = round(tp*1./(tp+fp),3)
        recall = round(tp*1./(tp+fn),3)
        f1=round(2*(precision*recall)/(precision+recall),3)
        capsnet_acc=acc

        diff_accs.append([capsnet_acc-lr_acc])
    return np.array(diff_accs)


# -------- markdown --------
# # genetic algorithm to find solution


class MyProblem(ea.Problem):
    def __init__(self):
        name = 'MyProblem'
        M = 1                               #
        maxormins = [-1]
        #Dim = 18265
        Dim = len([line for line in open(input_gene_file)])
        varTypes = [1] * Dim
        lb = [0] *Dim                       #
        ub = [1] *Dim
        lbin = [1] *Dim
        ubin = [1] *Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Vars = np.array(pop.Phen,dtype='int') #
        pop.ObjV=compute_diff_acc(Vars)



problem = MyProblem()
Encoding = 'RI'
# Encoding = 'BG'
NIND = 30
#NIND = 3
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)                                        # 创建区域描述器
population = ea.Population(Encoding, Field, NIND)

# myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)
myAlgorithm = ea.soea_SEGA_templet(problem, population)
myAlgorithm.MAXGEN = 200
#myAlgorithm.MAXGEN = 2
myAlgorithm.mutOper.F = 0.6
myAlgorithm.mutOper.Pm = 0.2
myAlgorithm.recOper.XOVR = 0.9
myAlgorithm.logTras = 1
myAlgorithm.verbose = True
myAlgorithm.drawing = 0


###
[BestIndi, population] = myAlgorithm.run()
BestIndi.save()



np.sum(BestIndi.Phen)
best_genes=np.array(BestIndi.Phen,dtype='int')
print('number of best genes:{}'.format(np.sum(best_genes)))
best_genes.shape

np.save('best_genes.iter{}.npy'.format(iter_k),best_genes)

##########
def compute_train_test_acc(gene_matrix,input_gene_file='sorted_genes2cij.pc5.top1200'):
    #(population_size, gene_set_size)
    all_input_genes=[]
    with open(input_gene_file) as fr:
        for line in fr:
            all_input_genes.append(line.strip())
    diff_accs=[]
    for j in range(gene_matrix.shape[0]):
        gene_vector=gene_matrix[j]
#         print('gene_vector:{}'.format(gene_vector))
        target_genes={}
        target_genes={gene:1 for k,gene in enumerate(all_input_genes) if gene_vector[k]==1}
#         print('target genes number:{}'.format(len(target_genes)))

        #compute acc
        x_train=X.iloc[train_idx] #downsample control such that balanced train case:control
        x_test=X.iloc[te_idx]
        x_test=np.array(x_test)
        y_test =Y[te_idx]
        x_train=np.array(x_train)
        y_train =Y[train_idx]

        for i in range(X.shape[1]):
            if X.columns[i].split(':')[1] not in target_genes: #there are some duplicated genes located on different chrs
                x_train[:,i]=0.
                x_test[:,i]=0.

        #LR
        y_pred = logisticRegr.predict(x_train)
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        lr_acc_train = round((tp + tn) * 1. / (tp + fp + tn + fn),3)

        y_pred = logisticRegr.predict(x_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        lr_acc_test = round((tp + tn) * 1. / (tp + fp + tn + fn),3)

        #Capsnet
        x_train=x_train.reshape(x_train.shape[0],1,1,x_train.shape[1])
        x_test=x_test.reshape(x_test.shape[0],1,1,x_test.shape[1])

        k=int(x_train.shape[0]/1000)+1
        y_pred=np.array([])
        for i in range(k):
            upper = min(x_train.shape[0],(i+1)*1000)
        #     print(i,upper)
            in_train=Variable(torch.tensor(x_train[(i*1000):upper]).to(device))
            y_pred0 = model(in_train).detach().cpu().numpy()
            y_pred0 = np.argmax(y_pred0, axis=1)
            y_pred = np.concatenate([y_pred,y_pred0])
        y_true = y_train

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = round((tp + tn) * 1. / (tp + fp + tn + fn),3)
        precision = round(tp*1./(tp+fp),3)
        recall = round(tp*1./(tp+fn),3)
        f1=round(2*(precision*recall)/(precision+recall),3)
        capsnet_acc_train=acc

        in_test=Variable(torch.tensor(x_test).to(device))
        y_pred = model(in_test).detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        y_true = y_test
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        capsnet_acc_test = round((tp + tn) * 1. / (tp + fp + tn + fn),3)

        print('PCA-LR train acc:{}\tCapsnet train acc:{}\tDiff:{}\n'.
              format(lr_acc_train,capsnet_acc_train,round(capsnet_acc_train-lr_acc_train,3)))
        print('PCA-LR test acc:{}\tCapsnet test acc:{}\tDiff:{}\n'.
              format(lr_acc_test,capsnet_acc_test,round(capsnet_acc_test-lr_acc_test,3)))
    return



compute_train_test_acc(best_genes,input_gene_file)

best_genes_array=np.load('best_genes.iter{}.npy'.format(iter_k))
input_genes=[line.strip() for line in open(input_gene_file)]
maxdiffacc_genes=[input_genes[i] for i,j in enumerate(best_genes_array[0]) if j==1]
len(maxdiffacc_genes)
with open('best_genes.iter{}.list'.format(iter_k),'w') as fw:
    fw.write('\n'.join(maxdiffacc_genes)+'\n')

print('done')

