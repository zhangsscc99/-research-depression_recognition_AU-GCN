import torch.nn as nn
import torch.nn.functional as F
from pygcn_copy.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        #self.gc2 = nn.Linear(nhid, 1)
        self.linear = nn.Linear(nclass, 1)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        #return F.log_softmax(x, dim=1)
        return self.linear(x)

        #所以gcn确实是这样，只需要改一层就可以了，输出层就OK。一行行源码进行解构
