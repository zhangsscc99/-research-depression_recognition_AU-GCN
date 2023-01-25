from __future__ import division
from __future__ import print_function


import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

#from pygcn.adj_matrix import get_all
#from pygcn.adj_matrix import adj_matrix

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

from pandas import *
 
# reading CSV file
data = read_csv("/Users/mac/Desktop/AUGCN/pygcn/205_2_Northwind_video.csv")
 
# converting column data to list
AU1 = data['AU01_c'].tolist()
#AU1, AU2, AU4）mouth  AU10, AU12, AU14, AU15, AU17, and AU25
AU2=data['AU02_c'].tolist()

AU4=data['AU04_c'].tolist()

AU10=data['AU10_c'].tolist()
AU12=data['AU12_c'].tolist()
AU14=data['AU14_c'].tolist()

AU15=data['AU15_c'].tolist()
AU17=data['AU17_c'].tolist()
AU25=data['AU25_c'].tolist()
AU_lst=[AU1,AU2,AU4,AU10,AU12,AU14,AU15,AU17,AU25]





# %%
lst=[]
for i in range(9):
    print(int(AU_lst[i][0]))
    lst.append(int(AU_lst[i][0]))

# %%
AU_set_lst=[]
for i in range(len(AU_lst[0])):
    set=[]
    for j in range(len(AU_lst)):
        set.append(AU_lst[j][i])
    AU_set_lst.append(set)
    


# %%
len(AU_set_lst)

# %%
import torch
AU_set_lst = torch.LongTensor(AU_set_lst)
#AU_set_lst = AU_set_lst.to(torch.float32)
embedding = torch.nn.Embedding(num_embeddings=9, embedding_dim=40)

# %%
AU_set_lst

# %%
features=embedding(AU_set_lst[0])
adj2=[0]*9
adj3=[]
for i in range(9):
    adj3.append(adj2)
adj=torch.FloatTensor(adj3)


#10 sets of AU_data: 0,1,1,1,....


def adj_matrix(AU_inc1,AU_inc2,feature):
    cnt1=0
    cnt2=0
    cnt_joint=0

    
    

    for i in range(len(feature)):
        if  feature[i][AU_inc1]==1.0:
            cnt1+=1
        if  feature[i][AU_inc2]==1.0:
            cnt2+=1
        if  feature[i][AU_inc2]==1.0 and feature[i][AU_inc1]==1.0:
            cnt_joint+=1
    

    AU1_AU2_joint_count = cnt_joint # Number of instances where both AU1 and AU4 are present
    AU2_count = cnt2 # Number of instances where AU4 is present
    total_count = len(feature) # Total number of instances in the dataset

    P_AU1_AU2 = AU1_AU2_joint_count / total_count
    P_AU2 = AU2_count / total_count
    #P_AU1_given_AU2 = P_AU1_AU2 / P_AU2
    if AU2_count==0.0:
        return 0.0
    P12=AU1_AU2_joint_count/AU2_count
    return P12
"""
index_pair=[]
for i in range(9):
    for j in range(9):
        index_pair.append([i,j])
print(index_pair,len(index_pair))
"""
def get_all():
    res=[]

    for i in range(9):
        for j in range(9):
            res.append(adj_matrix(i,j,AU_set_lst))
    import numpy as np                  #导入numpy模块，并重命名为np
    x = np.array(res)     #x是一维数组 
    d = x.reshape((9,9))                #将x重塑为2行4列的二维数组
    return d

adj=get_all()

adj=torch.FloatTensor(adj)
print(adj)




#adj=adj.long()
idx_train=[0]
idx_val=[0]


import copy
#features=copy.deepcopy(features_AU)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            #nclass=labels.max().item() + 1,
            nclass=1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

#我设置的loss 这个是
loss_func = torch.nn.MSELoss()

labels=[[100]]*9
labels = torch.LongTensor(labels)
labels = labels.to(torch.float32)



# %% [markdown]
# # dataset preparation

# %%



def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    #print(output[idx_train].shape)
    #print(labels.shape)
    #print(labels[idx_train].shape)
    #print(output)
    #print(labels)
    #loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train=loss_func(output[idx_train], labels[idx_train])
    
    #acc_train = accuracy(output[idx_train], labels[idx_train])

    
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    loss_val=loss_func(output[idx_val], labels[idx_val])
    #acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          #'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          #'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    print(output.shape)
    #loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    loss_test=loss_func(output[idx_train], labels[idx_train])
    #acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))
train(50)
test()


