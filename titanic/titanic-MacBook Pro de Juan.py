import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#extract data from data set 
data_valid= pd.read_csv('test.csv')
data_train= pd.read_csv('train.csv')

#data train cleanning 
for i in range(len(data_train)):
    #sex mod
    if data_train['Sex'][i] == 'male':
        data_train['Sex'][i] =0
    if data_train['Sex'][i] == 'female':
        data_train['Sex'][i] =1
    #Embarked mod
    if data_train['Embarked'][i] == 'C':
        data_train['Embarked'][i] =0
    if data_train['Embarked'][i] == 'Q':
        data_train['Embarked'][i] =1
    if data_train['Embarked'][i] == 'S':
        data_train['Embarked'][i] =2
    #cabin mod 
    if pd.isnull(data_train['Cabin'][i]) == True:
        data_train['Cabin'][i]=0
    else:
        data_train['Cabin'][i]=1
    
#data valid cleanning 
for i in range(len(data_valid)):
    #sex mod
    if data_valid['Sex'][i] == 'male':
        data_valid['Sex'][i] =0
    if data_valid['Sex'][i] == 'female':
        data_valid['Sex'][i] =1
    #Embarked mod
    if data_valid['Embarked'][i] == 'C':
        data_valid['Embarked'][i] =0
    if data_valid['Embarked'][i] == 'Q':
        data_valid['Embarked'][i] =1
    if data_valid['Embarked'][i] == 'S':
        data_valid['Embarked'][i] =2
    #cabin mod 
    if pd.isnull(data_valid['Cabin'][i]) == True:
        data_valid['Cabin'][i]=0
    else:
        data_valid['Cabin'][i]=1

#new variables 
data_train['Family_size']=data_train['SibSp']+data_train['Parch']

#split data
items=['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked','Family_size']
X=data_train[items]
y=data_train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#Model 
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train,early_stopping_rounds=5,eval_set=[(X_valid, y_valid)],verbose=False)

