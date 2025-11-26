import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cdlib
from cdlib import algorithms
from cdlib import viz
import time

S1=np.load("./train_Matrix/flickr_img_S.npy")
S2=np.load("./train_Matrix/flickr_text_S.npy")
S=S1+0.01*S2
S_ = np.where(S > 0.75, 1, 0)
n = S_.shape[0]  # 获取矩阵的维度  
for i in range(n):  
    S_[i, i] = 0 

rows, cols = np.where(S_ == 1)
edges = zip(rows.tolist(), cols.tolist())
graph = nx.Graph()
graph.add_edges_from(edges)



leiden_coms = algorithms.leiden(graph)
#print(leiden_coms.to_json())

community=[]
for i in leiden_coms.communities:
  community.append(i)
  
print(community)