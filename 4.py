import pandas as pd
from scipy import sparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#
df=pd.read_csv("large_twitch_edges.csv")
p=np.unique( df["numeric_id_1"].tolist() + df["numeric_id_2"].tolist() )
source=df["numeric_id_1"].tolist()
target=df["numeric_id_2"].tolist()
n = len(p)
L=len(target)

#
data=np.ones(L)
M=sparse.csr_matrix((data,(source,target)),(n , n))
M=sparse.triu(M,k=1)

#
M= M + M.T
k=M.sum(axis=0)
#
k.min()
#
k.mean()
#
k.max()
#
c, b = np.histogram(np.array(k)[0],bins=10**np.arange(0,4,0.1))
b= (b[1:] + b[:-1])/2
#
plt.plot(b,c,"+")
#
plt.loglog(b, c,'+')

import sys
import gzip
from random import random, seed, shuffle

class Graph(object):
        pass


def shuffleEdges(myList):
        """iterator over shuffled list elements"""
        shuffle(myList)
        ctr = 0
        for x in myList:
            ctr += 1
            yield ctr, x


# READ CFG 1{{{

def fetchGraph_eList(fName):
        """construct adjacengy list graph from edge list in DIMACS formatted file"""
        G     = Graph()
        eList = []
        myOpen = gzip.open if fName.split(".")[-1]=="gz" else open
        with myOpen(fName,'r') as f:
            for line in f:
                c = line.split()
                if c[0]=='p':
                        G.v   = int(c[2])
                        G.e   = int(c[3])
                        eType = int(c[4])

                elif c[0]=='e':
                        vi = int(c[1])
                        vj = int(c[2])
                        eList.append((vi,vj))

        G.eList = eList
        return G
# 1}}}

# UNION-FIND 1{{{

class UnionFind_byRank_pathCompression(object):
        """ union find data structure implementing union-by-rank 
        and path compression
        NOTE: 
        -# keeps track of largest cluster size
        -# keeps track of the number of components
        """
        def __init__(self):
                self.par   = dict()
                self.size  = dict()
                self.rank  = dict() 

                self.nComp = 0 
                self.cMax  = 0


        def newElement(self,i):
                """initialize new element"""
                self.par[i]  = i
                self.size[i] = 1
                self.rank[i] = 1
                self.nComp  += 1


        def find(self,i):
                """find root node of element and perform compression"""
                if self.par[i]!=self.par[self.par[i]]: 
                        self.par[i] = self.find(self.par[i])
                return self.par[i]

        
        def union(self,i,j):
                """union two elements by rank"""
                ii, jj = self.find(i), self.find(j)
                if ii != jj:

                        if self.rank[ii] > self.rank[jj]: 
                                ii,jj = jj,ii

                        if self.rank[ii] == self.rank[jj]: 
                                self.rank[jj]+=1

                        self.size[jj] += self.size[ii]
                        self.par[ii]   = jj

                        self.cMax   = max(self.size[jj],self.cMax)
                        self.nComp -=1

                        del self.size[ii]
# 1}}}


def main_bondPercolation_unionFind():

        bondFile = sys.argv[1]
        mySeed   = int(sys.argv[2])

        seed(mySeed)
        G   = fetchGraph_eList(bondFile)
        uf  = UnionFind_byRank_pathCompression()
        for i in range(G.v): uf.newElement(i)
        for nEdges,(i,j) in shuffleEdges(G.eList):
            uf.union(i,j)
            print "%4d %4d %4d %4d %4d"%(nEdges, G.v, G.e, uf.nComp, uf.cMax)


main_bondPercolation_unionFind()


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from random import random

def newman_ziff(p, L):
    N = L*L
    s = np.zeros((L, L))
    c = np.zeros((L, L))
    nc = 0
    for i in range(N):
        x, y = np.random.randint(0, L), np.random.randint(0, L)
        if random() < p:
            s[x, y] = 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if 0 <= x+dx < L and 0 <= y+dy < L and s[x+dx, y+dy]:
                    c1, c2 = c[x, y], c[x+dx, y+dy]
                    if c1 != c2:
                        if c1 == 0 and c2 == 0:
                            nc += 1
                            c[x, y], c[x+dx, y+dy] = nc, nc
                        elif c1 == 0:
                            c[x, y] = c2
                        elif c2 == 0:
                            c[x+dx, y+dy] = c1
                        else:
                            c[c == c2] = c1
    return c

p = np.linspace(0, 1, 100)
L = 100
ensemble = 100
cluster_sizes = []
for pi in p:
    cs = []
    for _ in range(ensemble):
        c = newman_ziff(pi, L)
        cs.append(np.bincount(c.flatten())[1:])
    cluster_sizes.append(np.mean(cs))
np.bincount(c.flatten()[1:])
plt.plot(p, cluster_sizes)
plt.xlabel('Occupation probability')
plt.ylabel('Average cluster size')
plt.show()


c.dtype


import numpy as np
import matplotlib.pyplot as plt
from random import random

def newman_ziff(p, L):
    N = L*L
    s = np.zeros((L, L))
    c = np.zeros((L, L))
    nc = 0
    for i in range(N):
        x, y = np.random.randint(0, L, size=2)
        if random() < p:
            s[x, y] = 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if 0 <= x+dx < L and 0 <= y+dy < L and s[x+dx, y+dy]:
                    c1, c2 = c[x, y], c[x+dx, y+dy]
                    if c1 != c2:
                        if c1 == 0 and c2 == 0:
                            nc += 1
                            c[x, y], c[x+dx, y+dy] = nc, nc
                        elif c1 == 0:
                            c[x, y] = c2
                        elif c2 == 0:
                            c[x+dx, y+dy] = c1
                        else:
                            c[c == c2] = c1
    return c

p = np.linspace(0, 1, 100)
L = 100
ensemble = 10
cluster_sizes = []
for pi in p:
    cs = []
    for _ in range(ensemble):
        c = newman_ziff(pi, L)
        cs.append(np.bincount(c.flatten())[1:])
    cluster_sizes.append(np.mean(cs))

plt.plot(p, cluster_sizes)
plt.xlabel('Occupation probability')
plt.ylabel('Average cluster size')
plt.show()


import matplotlib.pyplot as plt
import numpy as np

def calculate_plot_average_cluster_size(L):
    N = L*L    # Number of sites
    ensemble = 100   # Number of ensembles
    p_list = np.linspace(0.1, 1, 10)  # generate a list of occupation probabilities
    avg_cluster_size = []

    for p in p_list:
        avg_size = 0
        for i in range(ensemble):
            grid = np.random.rand(L, L) < p  # Initialize the lattice with random occupation

            # Perform the Newman-Ziff algorithm
            n_clusters, cluster_id = 0, np.zeros((L, L), dtype=int)
            for i in range(L):
                for j in range(L):
                    if grid[i][j]:
                        neighbors = cluster_id[i, j-1], cluster_id[i-1, j]
                        if 0 in neighbors:  # the site is unoccupied
                            if neighbors[1] == 0:
                                cluster_id[i][j] = neighbors[0]
                            elif neighbors[0] == 0:
                                cluster_id[i][j] = neighbors[1]
                            else:
                                n_clusters += 1
                                cluster_id[i][j] = n_clusters
                        else:
                            cluster_id[i][j] = max(neighbors)

            # Calculate the size of each cluster
            size = np.bincount(cluster_id.flatten())
            size[0] = 0  # exclude the background cluster
            avg_size += np.mean(size)

        avg_cluster_size.append(avg_size/ensemble)
    # Plot the result
    plt.plot(p_list, avg_cluster_size)
    plt.xlabel('Occupation probability')
    plt.ylabel('Average cluster size')
    plt.show()
calculate_plot_average_cluster_size(64)





import numpy as np
import matplotlib.pyplot as plt

def newman_ziff(L, p):
    # Initialize lattice with random occupation
    lattice = np.random.rand(L, L) < p
    clusters = np.zeros_like(lattice)
    cluster_sizes = []
    next_cluster_label = 2
    for x in range(L):
        for y in range(L):
            if lattice[x, y]:
                neighbors = clusters[max(x-1, 0):min(x+2, L), max(y-1, 0):min(y+2, L)]
                if not np.any(neighbors):
                    clusters[x, y] = next_cluster_label
                    next_cluster_label += 1
                else:
                    clusters[x, y] = np.unique(neighbors)[1]
    for label in range(2, next_cluster_label):
        cluster_sizes.append(np.sum(clusters == label))
    return np.mean(cluster_sizes)

# Plot average cluster size as a function of occupation probability
probabilities = np.linspace(0, 1, num=100)
sizes = [10, 20, 30, 40, 50]
for L in sizes:
    avg_sizes = [newman_ziff(L, p) for p in probabilities]
    plt.plot(probabilities, avg_sizes, label=f"L={L}")
plt.legend()
plt.xlabel("Occupation probability")
plt.ylabel("Average cluster size")
plt.show()
