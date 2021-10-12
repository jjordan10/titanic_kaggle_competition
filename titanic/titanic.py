


#data train cleanning 
#sex mod
data_train['Sex']=data_train['Sex'].replace(['male'],0)
data_train['Sex']=data_train['Sex'].replace(['female'],1)
#Age

#embarked mod
data_train['Embarked']=data_train['Embarked'].replace(['C'],0)
data_train['Embarked']=data_train['Embarked'].replace(['Q'],1)
data_train['Embarked']=data_train['Embarked'].replace(['S'],2)
#cabin mod
for i in range(len(data_train)):
    if pd.isnull(data_train['Cabin'][i]) == True:
        data_train['Cabin'][i]=0
    else:
        data_train['Cabin'][i]=1
data_train['Cabin']= data_train['Cabin'].astype('int64', copy=False)
  

#data valid cleanning 
#sex mod
data_valid['Sex']=data_valid['Sex'].replace(['male'],0)
data_valid['Sex']=data_valid['Sex'].replace(['female'],1)
#embarked mod
data_valid['Embarked']=data_valid['Embarked'].replace(['C'],0)
data_valid['Embarked']=data_valid['Embarked'].replace(['Q'],1)
data_valid['Embarked']=data_valid['Embarked'].replace(['S'],2)
#cabin mod
for i in range(len(data_valid)):
    if pd.isnull(data_valid['Cabin'][i]) == True:
        data_valid['Cabin'][i]=0
    else:
        data_valid['Cabin'][i]=1
data_valid['Cabin']= data_valid['Cabin'].astype('int64', copy=False)


#new variables 
data_train['Family_size']=data_train['SibSp']+data_train['Parch']
data_valid['Family_size']=data_valid['SibSp']+data_valid['Parch']


#split data
items=['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked','Family_size']
X=data_train[items]
y=data_train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=45, test_size=0.3, shuffle=True)
X_valid=data_valid[items]


#Model
my_model = XGBRegressor(n_estimators=1500, learning_rate=0.01, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_test, y_test)], 
             verbose=False)


# Get predictions
predictions = my_model.predict(X_valid) # Your code here
predictions = predictions.round()

kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(my_model, X_test, y_test, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

submit=pd.DataFrame({'PassengerId':data_valid['PassengerId'],'Survived':predictions})
submit.to_csv('submit.csv',index=False)

accuracy = accuracy_score(y_test, predictions)
print(accuracy)


# Calculate MAE
#mae_2 = mean_absolute_error(predictions_2,y_valid) # Your code here


#my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
#my_model.fit(X_train, y_train,early_stopping_rounds=5,verbose=False)

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


#extract data from data set 
data_test= pd.read_csv('test.csv')
data_train= pd.read_csv('train.csv')

y = data_train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(data_train[features])
X_test = pd.get_dummies(data_test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

#Model
my_model = XGBRegressor(n_estimators=3000, learning_rate=0.01)
my_model.fit(X, y,  
             verbose=False)

kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(my_model, X, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

output = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, y)
Y_pred = random_forest.predict(X_test)
random_forest.score(X, y)
acc_random_forest = round(random_forest.score(X, y) * 100, 2)
acc_random_forest

kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(random_forest, X, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

output = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)