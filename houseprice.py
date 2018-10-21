#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 20:05:12 2018

@author: ljk
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2


global enclist,objlist,ohelist
def sol(csv):
    
    global csvcp
    usv=csv.shape[0]
    csvcp=csv.copy()
    csvcp=csvcp.dropna(axis=1,thresh=0.5*usv)
    u=csvcp.columns
    t=csvcp.dtypes
    objlist=['MSSubClass']
    for k in range(len(u)):
        if '64' not in str(t[k]):
            objlist.append(str(u[k]))
            csvcp.loc[:,u[k]]=csvcp.loc[:,u[k]].fillna(method='bfill').fillna(method='ffill')
        else:
            means=csvcp.loc[:,u[k]].mean()
            csvcp.loc[:,u[k]]=csvcp.loc[:,u[k]].fillna(means)
    for k in objlist:
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit(csvcp[k])
        integer_encoded=encoded.transform(csvcp[k])
        csvcp.loc[:,k]=integer_encoded
#        ohe=OneHotEncoder().fit(integer_encoded.reshape(-1,1))
#        tt=ohe.transform(integer_encoded.reshape(-1,1)).toarray()
#        tl=len(tt[0])
#        dt=pd.DataFrame(tt).set_index(csvcp.index)
#        for ts in range(tl):
#            uk=k+str(ts)
#            csvcp.loc[:,uk]=dt[ts]
#        csvcp.drop(k,axis=1,inplace=True)
    csvcp=csvcp.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    
    
#    PPP=PCA(n_components=0.5,whiten=True)
#    csvcp=PPP.fit_transform(csvcp)
    return csvcp

train=pd.read_csv('/home/ljk/下载/kaggle/all (2)/train.csv',index_col='Id')
test=pd.read_csv('/home/ljk/下载/kaggle/all (2)/test.csv',index_col='Id')

y=train['SalePrice']
trains=train.drop(['SalePrice'],axis=1)
u=pd.concat((trains,test),axis=0)
ut=sol(u)
li1=[i for i in test.index]
li2=[i for i in train.index]


#train1=ut[0:len(li2)]#ut.drop(li2)
#test1=ut[len(li2):]#ut.drop(li1)
#
test1=ut.drop(li2)
train1=ut.drop(li1)


#usb=SelectPercentile(chi2, percentile=70)
#usb.fit(train1, y)
#train1=usb.transform(train1)
#test1=usb.transform(test1)

from sklearn import tree
lr = tree.DecisionTreeRegressor(max_depth=8,max_features=53,random_state=2)
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



#best=[1,0,0,0]
#for i in range(1,10):
#    for h in range(1,290):
#        for j in range(1,10):
#                lr =tree.DecisionTreeRegressor(max_depth=i,max_features=h,random_state=j)
#                cross_predict = cross_val_predict(lr,train1,y,cv=5)
#                score=np.sqrt(metrics.mean_squared_error(np.log(y),np.log(cross_predict)))
#                if score<best[0]:
#                    best=[score,i,h,j]
#                print(i,h,j)