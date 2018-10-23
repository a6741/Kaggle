#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 22:31:16 2018

@author: ljk
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import re
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import OneHotEncoder




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#from sklearn.model_selection import train_test_split #废弃！！
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import BernoulliRBM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcess
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis








def anotherway(file):
    for k in file.index:
        if file.loc[(k,'Sex')]=='female':
            file.loc[(k,'Sex')]=0
        else:
            file.loc[(k,'Sex')]=1
        if file.loc[(k,'Embarked')]=='S':
            file.loc[(k,'Embarked')]=0
        elif file.loc[(k,'Embarked')]=='C':
            file.loc[(k,'Embarked')]=1
        else:
            file.loc[(k,'Embarked')]=2
        if len(re.findall('[a-z,A-Z]',file.loc[(k,'Ticket')]))>0:
            file.loc[(k,'Ticket')]=1
        else:
            file.loc[(k,'Ticket')]=0
    names=file['Name']
    namedi={'Miss':1,'Matser':2,'Mrs':3,'Mr':4}
    for q in names.index:
        named=(names[q].split(',')[1]).split('.')[0].replace(' ','')
        if named in namedi.keys():
            names.loc[q]=namedi[named]
        else:
            names.loc[q]=5
    file.drop(['Name','Cabin','Ticket'],axis=1,inplace=True)
    file.loc[:,'Name']=names
    file.loc[:,'Age']=file.loc[:,'Age'].fillna(0)
    
    fnu=file['Fare'].isnull()
    fm=file['Fare'].mean()
    for k in file.index:
        if fnu[k]:
            file.loc[(k,'Fare')]=fm
#    for f in file['Fare'].index:
#        if file.loc[(f,'Fare')]!=0:
#            file.loc[(f,'Fare')]=np.log(file.loc[(f,'Fare')])
#        else:
#            file.loc[(f,'Fare')]=0
    ma=file.groupby(['Name'])['Age'].mean()
    mnu=ma.isnull()
    for k in ma.index:
        if mnu[k]:
            p=k
            isout=1
            fla=True
            while fla:
                if p>ma.index.max():
                    isout=2
                    la=0
                    break
                try:
                    la=ma[p+1]
                    if la==0:                        
                        fla=False
                    else:
                        p+=1
                except:
                    p+=1
            p=k
            fla=True
            while fla:
                if p<ma.index.min():
                    isout=2
                    ne=0
                    break
                try:
                    ne=ma[p-1]
                    if ne==0:
                        fla=False
                    else:
                        p-=1
                except:
                    p-=1        
            ma[k]=(la+ne)*isout/2
    isnull=file['Age'].isnull()
    file.loc[:,'Family']=file.loc[:,'SibSp']+file.loc[:,'Parch']
    for k in isnull.index:
        if isnull[k]:
            file.loc[(k,'Age')]=ma[file.loc[(k,'Name')]]
#    for tt in file['Age'].index:
#        agn=file.loc[(tt,'Age')]
#        if agn<16:
#            file.loc[(tt,'Age')]=1
#        elif agn<50:
#            file.loc[(tt,'Age')]=2
#        else:
#            file.loc[(tt,'Age')]=3
#    for tt in file['Age'].index:
#        ag=file.loc[(tt,'Age')]
#        fp=file.loc[(tt,'Parch')]
#        if ag<20 and fp>0:
#            file.loc[(tt,'Type')]=0
#        elif ag>20 and fp==0:
#            file.loc[(tt,'Type')]=1
#        elif ag<20 and fp==0:
#            file.loc[(tt,'Type')]=2
#        else:
#            file.loc[(tt,'Type')]=3
    file.drop(['SibSp','Parch'],axis=1,inplace=True)
#    namee=['Name','Embarked']
#    for sbs in namee:
#        oh=OneHotEncoder().fit_transform(file[sbs].reshape(-1,1)).toarray()
#        ohl=oh.shape[1]
#        doh=pd.DataFrame(oh).set_index(file.index)
#        num=sbs
#        for ti in range(ohl):
#            file.loc[:,num+str(ti)]=doh[ti]
#    file.drop(namee,axis=1,inplace=True)
    #file=file.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    #file.drop(['Name'],axis=1,inplace=True)
    return file
def ybm(file):
#    for k in file.index:
#        if file.loc[(k,'Sex')]=='female':
#            file.loc[(k,'Sex')]=0
#        else:
#            file.loc[(k,'Sex')]=1
#        if file.loc[(k,'Embarked')]=='S':
#            file.loc[(k,'Embarked')]=0
#        elif file.loc[(k,'Embarked')]=='C':
#            file.loc[(k,'Embarked')]=1
#        else:
#            file.loc[(k,'Embarked')]=2
    names=file['Name']
    namedi={'Miss':1,'Matser':2,'Mrs':3,'Mr':4}
    for q in names.index:
        named=(names[q].split(',')[1]).split('.')[0].replace(' ','')
        if named in namedi.keys():
            names.loc[q]=named#i[named]
        else:
            names.loc[q]='other'#5
    
    file.drop(['Name','Cabin','Ticket'],axis=1,inplace=True)
    file.loc[:,'Name']=names
    file.loc[:,'Age']=file.loc[:,'Age'].fillna(0)
    
    fnu=file['Fare'].isnull()
    fm=file['Fare'].mean()
    for k in file.index:
        if fnu[k]:
            file.loc[(k,'Fare')]=fm
#    for f in file['Fare'].index:
#        if file.loc[(f,'Fare')]!=0:
#            file.loc[(f,'Fare')]=np.log(file.loc[(f,'Fare')])
#        else:
#            file.loc[(f,'Fare')]=0
    ma=file.groupby(['Name'])['Age'].mean()
    mnu=ma.isnull()
    for k in ma.index:
        if mnu[k]:
            p=k
            isout=1
            fla=True
            while fla:
                if p>ma.index.max():
                    isout=2
                    la=0
                    break
                try:
                    la=ma[p+1]
                    if la==0:                        
                        fla=False
                    else:
                        p+=1
                except:
                    p+=1
            p=k
            fla=True
            while fla:
                if p<ma.index.min():
                    isout=2
                    ne=0
                    break
                try:
                    ne=ma[p-1]
                    if ne==0:
                        fla=False
                    else:
                        p-=1
                except:
                    p-=1        
            ma[k]=(la+ne)*isout/2
    isnull=file['Age'].isnull()
    file.loc[:,'Family']=file.loc[:,'SibSp']+file.loc[:,'Parch']
    for k in isnull.index:
        if isnull[k]:
            file.loc[(k,'Age')]=ma[file.loc[(k,'Name')]]
    
#    for tt in file['Age'].index:
#        ag=file.loc[(tt,'Age')]
#        fp=file.loc[(tt,'Parch')]
#        if ag<20 and fp>0:
#            file.loc[(tt,'Type')]=0
#        elif ag>20 and fp==0:
#            file.loc[(tt,'Type')]=1
#        elif ag<20 and fp==0:
#            file.loc[(tt,'Type')]=2
#        else:
#            file.loc[(tt,'Type')]=3
    file.drop(['SibSp'],axis=1,inplace=True)
    
    file=pd.get_dummies(file)

#    namee=['Name','Embarked']
#    for sbs in namee:
#        oh=OneHotEncoder().fit_transform(file[sbs].reshape(-1,1)).toarray()[:,1:]
#        ohl=oh.shape[1]
#        doh=pd.DataFrame(oh).set_index(file.index)
#        num=sbs
#        for ti in range(ohl):
#            file.loc[:,num+str(ti)]=doh[ti]

    
    
#    file.drop(namee,axis=1,inplace=True)
    file=file.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    #file.drop(['Name'],axis=1,inplace=True)
    return file
def csv(file):
    for k in file.index:
        if file.loc[(k,'Sex')]=='female':
            file.loc[(k,'Sex')]=0
        else:
            file.loc[(k,'Sex')]=1
        if file.loc[(k,'Embarked')]=='S':
            file.loc[(k,'Embarked')]=0
        elif file.loc[(k,'Embarked')]=='C':
            file.loc[(k,'Embarked')]=1
        else:
            file.loc[(k,'Embarked')]=2
    isnull=file['Cabin'].isnull()
    file.drop(['Name','Cabin'],axis=1,inplace=True)
    num=0
    for k in isnull.index:
        if not isnull[k]:
            num+=1
            #print(file.loc[(k,'Cabin')])
            #file.loc[(k,'Ticket')]
    t2=file['Age']
    t3=file['Ticket']
    for q in t3.index:
        qs=t3[q].split(' ')
        if len(qs)>1 or t3[q]=='LINE':
            tags=qs[0].replace('.','').replace('/','')
            tag=0
            for ppp in map(ord,tags):
                tag+=int(ppp)
            t3.loc[q]=tag
    file.loc[:,'Age']=file.loc[:,'Age'].fillna(0)
    
    file.loc[:,'Ticket']=file.loc[:,'Ticket'].astype('int')
    file=file.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    
    nums=KMeans(n_clusters=50,random_state=170).fit_predict(t3.reshape(-1,1))
    file.loc[:,'Ticket']=nums
    file.drop(['Age'],axis=1,inplace=True)
    fnu=file['Fare'].isnull()
    fm=file['Fare'].mean()
    for k in file.index:
        if fnu[k]:
            file.loc[(k,'Fare')]=fm
    file=file.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    nume=KMeans(n_clusters=num,random_state=170).fit_predict(file)
    file.loc[:,'Cabin']=nume
    file.loc[:,'Age']=t2
    ma=file.groupby(['Cabin'])['Age'].mean()
    mnu=ma.isnull()
    for k in ma.index:
        if mnu[k]:
            p=k
            isout=1
            fla=True
            while fla:
                if p>ma.index.max():
                    isout=2
                    la=0
                    break
                try:
                    la=ma[p+1]
                    if la==0:                        
                        fla=False
                    else:
                        p+=1
                except:
                    p+=1
            p=k
            fla=True
            while fla:
                if p<ma.index.min():
                    isout=2
                    ne=0
                    break
                try:
                    ne=ma[p-1]
                    if ne==0:
                        fla=False
                    else:
                        p-=1
                except:
                    p-=1        
            ma[k]=(la+ne)*isout/2
    isnull=file['Age'].isnull()
    for k in isnull.index:
        if isnull[k]:
            file.loc[(k,'Age')]=ma[file.loc[(k,'Cabin')]]
    file=file.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return file
#file.loc[:,'Group']=KMeans(n_clusters=50,random_state=170).fit_predict(file)
file = pd.read_csv('/home/ljk/下载/kaggle/all/train.csv',encoding='utf8',index_col='PassengerId')
tfi=pd.read_csv('/home/ljk/下载/kaggle/all/test.csv',encoding='utf8',index_col='PassengerId')


y=file['Survived']
files=file.drop(['Survived'],axis=1)
u=pd.concat((files,tfi),axis=0)
ano=ybm(u)

li1=[i for i in files.index]
li2=[i for i in tfi.index]


file=ano.drop(li2)
tfi=ano.drop(li1)

#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#usb=SelectKBest(chi2, k=5)
#usb.fit(file, y)
#file=usb.transform(file)

#from sklearn.decomposition import PCA

#from sklearn.preprocessing import PolynomialFeatures

#poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
#X_ploly = poly.fit_transform(file)

#from sklearn.tree import DecisionTreeClassifier
#cla2=DecisionTreeClassifier(max_depth=7)

#from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier
cla=RandomForestClassifier(max_depth=5, n_estimators=8,max_features=9,random_state=5)

#fibf.dropna()
#from sklearn.linear_model import LogisticRegression
#cla=LogisticRegression()

#from sklearn.ensemble import GradientBoostingClassifier
#cla2=GradientBoostingClassifier(n_estimators=30, learning_rate=0.9,max_depth=100, random_state=0)
#cla2.fit(file,y)

#shape=file.shape[0]
#file=cla2.apply(file).reshape(shape,-1)

#PPP=PCA(n_components='mle',whiten=True,svd_solver='full')
#PPP.fit(file)

#file=PPP.transform(file)

#from sklearn import linear_model

#cla = linear_model.LinearRegression()

#from sklearn.neighbors import KNeighborsClassifier
#cla = KNeighborsClassifier()

#from sklearn.naive_bayes import GaussianNB
#cla=GaussianNB()

#from sklearn.svm import LinearSVC
#cla=LinearSVC()

#cla=QuadraticDiscriminantAnalysis()

cla.fit(file,y)

#tfi=anotherway(tfi)
tfic=tfi.copy()
#tfic=usb.transform(tfic)




#shape=tfic.shape[0]
#tfic=cla2.apply(tfic).reshape(shape,-1)
#tfic=PPP.transform(tfic)

#poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
#tfic_ploly = poly.fit_transform(tfic)

tfi.loc[:,'pre']=cla.predict(tfic)
tu=tfi.loc[:,'pre'].astype('int')
tu.to_csv('pr.csv')

from sklearn.model_selection import cross_val_score
accu = cross_val_score(cla, file, y, cv=3, scoring="accuracy" ).mean()
score=cla.score(file,y)
#去掉sex的哑编码试试

#clf = SVC(kernel='linear',C=0.4)
#clf.fit(file,y)
#file = pd.read_csv('/home/ljk/下载/kaggle/all/test.csv',encoding='utf8',index_col='PassengerId')
#
#pred_y = clf.predict(test_x)
#
#print(classification_report(test_y,pred_y))





import threading

def testpara():
    global best
    best=[]
    ts=[]
    for i in range(1,10):
        for h in range(1,10):
            for j in range(1,10):
                for k in range(10):
                    t=threading.Thread(target=test,args=(i,h,j,k,))
                    t.start()
                    ts.append(t)
    for k in ts:
        k.join()
def test(i,h,j,k):
    global best
    cla=RandomForestClassifier(max_depth=i, n_estimators=h, max_features=j,random_state=k)
    accu = cross_val_score( cla, file, y, cv=3, scoring="accuracy" ).mean()
    best.append([accu,i,h,j,k])
    print(i,h,j,k)
    


#best=[0,0,0,0,0]
#for i in range(1,10):
#    for h in range(1,10):
#        for j in range(1,10):
#            for k in range(10):
#                cla=RandomForestClassifier(max_depth=i, n_estimators=h, max_features=j,random_state=k)
#                accu = cross_val_score( cla, file, y, cv=3, scoring="accuracy" ).mean()
#                if accu>best[0]:
#                    best=[accu,i,h,j,k]
#                print(i,h,j,k)

