# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_data=pd.read_csv("train - train.csv")
X=train_data.iloc[:, :-1].values
df_X = pd.DataFrame(X)
Y=train_data.iloc[:, 8].values
df_Y=pd.DataFrame(Y)

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
imputer=imputer.fit(X[:, 3:4])
X[:, 3:4]=imputer.fit_transform(X[:, 3:4])


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X= LabelEncoder()
X[:, 2]=labelencoder_X.fit_transform(X[:, 2])

labelencoder_X1= LabelEncoder()
X[:, 7]=labelencoder_X1.fit_transform(X[:, 7])


#test data processing
test_data=pd.read_csv("test - test (1).csv")
Z=test_data.iloc[:, ].values
df_testZ = pd.DataFrame(Z)

labelencoder_test= LabelEncoder()
Z[:, 2]=labelencoder_test.fit_transform(Z[:, 2])

labelencoder_X1= LabelEncoder()
Z[:, 7]=labelencoder_X1.fit_transform(Z[:, 7])

imputer_Z=Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
imputer_Z=imputer_Z.fit(Z[:, 3:4])
Z[:, 3:4]=imputer_Z.fit_transform(Z[:, 3:4])

imputer_Z1=Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
imputer_Z1=imputer_Z1.fit(Z[:, 6:7])
Z[:, 6:7]=imputer_Z.fit_transform(Z[:, 6:7])


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,Y)

Z_pred=regressor.predict(Z)

#graph

plt.scatter(Z[:,0],Z_pred,color="red")
plt.plot(X[:, 0], regressor.predict(X) ,color="cyan" )
plt.plot(Z[:, 0], Z_pred ,color="blue" )

plt.title("Survival Expectancy of Titanic (Cyan = Given data , Blue = Predicted Data)")

plt.xlabel("Passenger Id")
plt.ylabel("Survival (0,1)")
plt.show()






