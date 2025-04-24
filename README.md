# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Navinkumar v
RegisterNumber:  212223230141
*/
import pandas as pd
from sklearn.metrics import r2_score
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
r2= r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
## Dataset
![image](https://github.com/user-attachments/assets/5d89491a-56e7-47d3-b69c-d6ca3b52e990)
## Info
![image](https://github.com/user-attachments/assets/965425d7-68ad-4c65-9e9e-8c30993f76fe)
## NULL value
![image](https://github.com/user-attachments/assets/d1d3d325-2719-4bc6-be51-c1f21981cfca)
## Encoded
![image](https://github.com/user-attachments/assets/d70cfc95-d63b-4db0-b608-e20f1add2422)
## x and y value
![image](https://github.com/user-attachments/assets/f2c61c24-4ad9-4a41-b1b7-278117baf276)
![image](https://github.com/user-attachments/assets/6790474d-844f-4184-ae1f-286548036118)
## Algorithm
![image](https://github.com/user-attachments/assets/a3e7a361-256e-4a1d-a140-ba8b753cdfa9)
## R2 Score
![image](https://github.com/user-attachments/assets/340fdfa9-9e5d-476d-9a34-f8b833b0463a)
## Predicted
![image](https://github.com/user-attachments/assets/bb274489-d14d-45d5-93cc-92e4d71e0227)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
