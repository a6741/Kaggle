#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 20:05:12 2018

@author: ljk
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2

def findnoise(csv,trmax):
    thresh=5
    noiselis=set()
    for qs in csv:
        q=csv[qs]
        if '64' in str(q.dtype):
            mean=q.mean()
            std=q.std()
            for k in q.index:
                if k<=trmax:
                    if q.loc[k]<mean-thresh*std or q.loc[k]>mean+thresh*std:
                        noiselis.add(k)
    return noiselis
def sol(csv,trmax):
    global csvcp,colli
    usv=csv.shape[0]
    csvcp=csv.copy()
    csvcp=csvcp.dropna(axis=1,thresh=0.5*usv)#非空值少于90%就删去
    colli=findnoise(csvcp,trmax)
    csvcp.drop(colli,axis=0,inplace=True)
    u=csvcp.columns
    t=csvcp.dtypes
    objlist=['MSSubClass']
    delist=[]
    for k in range(len(u)):
        #print((csvcp[u[k]]).value_counts())
        mostv=(csvcp[u[k]]).value_counts()
        if mostv.iloc[0]>0.95*usv:
            delist.append(u[k])
        if '64' not in str(t[k]):# or k=='MSSubClass':
            objlist.append(str(u[k]))
            csvcp.loc[:,u[k]]=csvcp.loc[:,u[k]].fillna(mostv.index[0])
        else:
            means=csvcp.loc[:,u[k]].mean()#mode()[0]
            csvcp.loc[:,u[k]]=csvcp.loc[:,u[k]].fillna(means)
#    csvcp.drop(colli,axis=0,inplace=True)
#    csvcp.drop(delist,axis=1,inplace=True)
    csvcp=pd.get_dummies(csvcp)
    csvcp=pd.get_dummies(data=csvcp,columns=['MSSubClass'])
    csvcp=csvcp.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    
#    PPP=PCA(n_components=0.5,whiten=True)
#    csvcp=PPP.fit_transform(csvcp)
    return csvcp

train=pd.read_csv('/home/ljk/下载/kaggle/all (2)/train.csv',index_col='Id')
test=pd.read_csv('/home/ljk/下载/kaggle/all (2)/test.csv',index_col='Id')

y=train['SalePrice']
trains=train.drop(['SalePrice'],axis=1)
u=pd.concat((trains,test),axis=0)
trmax=train.index[-1]
ut=sol(u,trmax)
li1=[i for i in test.index]
li2=[i for i in train.index]

li1=[]
li2=[]
for i in ut.index:
    if i <=trmax:
        li2.append(i)
    else:
        li1.append(i)
y.drop(colli,axis=0,inplace=True)

train1=ut[0:len(li2)]#ut.drop(li2)
test1=ut[len(li2):]#ut.drop(li1)
#
#test1=ut.drop(li2)
#train1=ut.drop(li1)


#usb=SelectPercentile(chi2, percentile=70)
#usb.fit(train1, y)
#train1=usb.transform(train1)
#test1=usb.transform(test1)

#from sklearn import tree

#lr= ensemble.AdaBoostRegressor()

#from sklearn import svm
#lr = svm.SVR()

#from sklearn import ensemble
lr = ensemble.GradientBoostingRegressor(n_estimators=1000)#这里使用100个决策树


#from sklearn.ensemble import BaggingRegressor
#lr = BaggingRegressor()

#from sklearn.tree import ExtraTreeRegressor
#lr = ExtraTreeRegressor()


#from sklearn import neighbors
#lr= neighbors.KNeighborsRegressor()
#lr = ensemble.RandomForestRegressor()#max_depth=7,max_features=143,random_state=2)
#7,151,8
#lr = linear_model.LinearRegression()
#y=train1['SalePrice']
#xx=train1.drop(['SalePrice'],axis=1)
lr.fit(train1,y)



test.loc[:,'SalePrice']=lr.predict(test1)


tt=test['SalePrice']


tt.to_csv('hpr.csv',header=True)
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
cross_predict = cross_val_predict(lr,train1,y,cv=5)
score=np.sqrt(metrics.mean_squared_error(np.log(y),np.log(cross_predict)))


#
#best=[1,0,0,0,0]
#for i in range(1,10):
#    for h in range(1,10):
#        for j in range(1,279):
#            for k in range(10):
#                cla=ensemble.RandomForestRegressor(max_depth=i, n_estimators=h, max_features=j,random_state=k)
#                cross_predict = cross_val_predict(lr,train1,y,cv=5)
#                score=np.sqrt(metrics.mean_squared_error(np.log(y),np.log(cross_predict)))
#                if score<best[0]:
#                    best=[score,i,h,j,k]
#                print(i,h,j,k)
#with open('sb.txt','w') as f:
#    f.write(str(best))