# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 21:33:01 2018

@author: Dipanjan De
~Comments are added according to convenience
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset=pd.read_csv("Fertility.csv")
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 9].values

#Encoding the result
from sklearn.preprocessing import LabelEncoder
labelencoder_X= LabelEncoder()
Y[:,]=labelencoder_X.fit_transform(Y[:,])

Y=Y.astype('float') #As sklearn couldntrecognize the type so we had to typecast


#test_train_split
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)
#prediction of values
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)

