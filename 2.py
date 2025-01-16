import fastf1 as ff1
import matplotlib.pyplot as plt 
import numpy as np

ff1.Cache.enable_cache('C:/Users/Sina Roshandell/Desktop/Tda/New folder (2)/data set')

session1 = ff1.get_session(2019, 'Monza', 'Q')
session2 = ff1.get_session(2019, 'Spa', 'Q')
session1.load()
session2.load()


def embedding(data, dt, tau, dim):
    steps = int(tau / dt)
    N = len(data)
    path = []
    for i in range(N - (dim - 1) * steps):
        path.append(np.array(data[i: i + dim * steps: steps]))
    return np.array(path)


import gudhi as gd
from ripser import ripser
from persim import plot_diagrams
from gudhi.representations import DiagramSelector
from gudhi.representations import BettiCurve




def GPD(driver,race,unit):
    session = ff1.get_session(2019,race,'Q')
    session.load()
    fast= session.laps.pick_driver(driver).pick_fastest()
    lcdm = fast.get_car_data()
    temlcdm=embedding(lcdm[unit],1,5,3)
    dgms=ripser(temlcdm)["dgms"]
    dgms0=dgms[0]
    dgms1=dgms[1]
    pdm0 = DiagramSelector(use=True, point_type='finite')(dgms0)
    pdm1 = DiagramSelector(use=True, point_type='finite')(dgms1)
    return pdm0,pdm1,list(lcdm[unit])

Rs=["Australia","Bahrain","Azerbaijan","Germany","Japan","Monza","Brazil"]

p0,p1=GPD("LEC","Monza","Throttle")
plot_diagrams(p1)

L=['16', '44', '77', '5', '3', '27', '55', '23', '18', '7', '99', '20', '26', '4', '10', '8', '11', '63', '88', '33']
Rs=["Australia","Bahrain","Azerbaijan","Monza"]
L=['LEC', 'HAM', 'BOT', 'VET', 'GAS']

fs=[]
dname=[]
rname=[]
ys=[]
for j in range(len(Rs)):
    for i in range(len(L)):
        ph0,ph1,RPM=GPD(L[i],Rs[j],"RPM")
        B=BettiCurve(resolution=50)(ph1)
        dname.append(L[i])
        rname.append(Rs[j])
        ys.append(j)
        for k in range(50):
            fs.append(B[k])


fs=np.array(fs)
fs=fs.reshape((20,50))


import pandas as pd 
df=pd.DataFrame(fs)
df["name"]=dname
df["rname"]=rname
df["ys"]=ys
df

import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

feature_df = df[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]]
feature_df
X = np.asarray(feature_df)
y = np.asarray(df['ys'])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn import svm
clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train) 

yhat = clf.predict(X_test)
yhat-y_test