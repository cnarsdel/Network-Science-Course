import numpy as np
import scipy.sparse as ss
import pandas as pd
import matplotlib.pyplot as plt 

n,p=1000,0.5
f=[0,1]
A=np.random.choice(f,(n,n),p=[1-p,p])


pp=np.arange(0,1,0.01)
y=(pp**4)+4*(pp**3)*(1-pp)
plt.plot(pp,y)
plt.plot(pp,pp)

plt.matshow(A)
plt.colorbar()

AA=A[:-1,:-1]+A[1:,:-1]+A[:-1,1:]+A[1:,1:]
AA[AA>=2]=1
AA=AA[::2,::2]

plt.matshow(AA)
plt.colorbar()
AA.mean()
# omid.choupanian@gmail.com
# parsabigdeli77@gmail.com
mean=[]
for i in pp:
    f=[0,1]
    A=np.random.choice(f,(n,n),p=[1-i,i])
    AA=A[:-1,:-1]+A[1:,:-1]+A[:-1,1:]+A[1:,1:]
    AA[AA>=2]=1
    AA=AA[::2,::2]
    mean.append(AA.mean())

plt.plot(mean) 
mean=np.array(mean)
np.where(mean==1)
pp[8015]




#######################################



n,p=100,0.6
f=[0,1]
A=np.random.choice(f,(n,n),p=[1-p,p])

from scipy.ndimage import measurements
plt.imshow(measurements.label(A)[0])

s=measurements.label(A)[0]
np.max(measurements.label(A)[0])
measurements.label(A)[1]


for i in range(1,measurements.label(A)[1]):
    s[s==i]=np.random.randint(100)+100

plt.matshow(s)
plt.colorbar()

def ll(n,p):
    f=[0,1]
    A=np.random.choice(f,(n,n),p=[1-p,p])
    color=np.arange(measurements.label(A)[1]+1)
    f=np.random.shuffle(color)
    plt.matshow(color[measurements.label(A)[0]],cmap="inferno")
    plt.colorbar()
    return measurements.label(A)[1]
s=[]
for i in pp:
    s.append(ll(100,i))
plt.plot(s)
for i in range(100):
    n=500
    p=0.6
    f=[0,1]
    A=np.random.choice(f,(n,n),p=[1-p,p])
    color=np.arange(measurements.label(A)[1]+1)
    f=np.random.shuffle(color)
    #plt.matshow(color[measurements.label(A)[0]],cmap="inferno")
    pi=set(measurements.label(A)[0][:,0]).intersection(set(measurements.label(A)[0][:,-1]))
    if len(pi)>=3:
        print(pi)
        break

SF=measurements.label(A)[0]
pi
SF
SF[SF!=33]=5
plt.matshow(SF)

plt.matshow(SF)
np.bincount(measurements.label(A)[0].flatten()).max()
pi
plt.matshow(color[measurements.label(A)[0]],cmap="inferno")
plt.matshow(color[measurements.label(A)[0]],cmap="inferno")
ll(100,0.6)
ll(1000,0.4)
ll(100,0.3)
ll(100,0.6)
ll(100,0.7)
ll(100,0.8)