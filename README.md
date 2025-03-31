# EXP -04 SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries – Load necessary Python libraries.
2.Load Data – Read the dataset containing house details.
3.Preprocess Data – Clean and split the data into training and testing sets.
4.Select Features & Target – Choose input variables (features) and output variables (house price, occupants).
5.Train Model – Use SGDRegressor() to train the model.
6.Make Predictions – Use the model to predict house price and occupants.
7.Evaluate Performance – Check accuracy using error metrics.
8.Improve Model – Tune settings for better accuracy.

## Program:

```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:Lathika .K
RegisterNumber:  212224230140
*/
import numpy as np
import pandas as pd
froa sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset=fetch_california_housing()

df-pd.DataFrame(dataset.data,columns-dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

X-df.drop(columns=["AveOccup','HousingPrice'])
Y=df[["'AveOccup', 'HousingPrice']]
Xtrain,X_test,Y_train,Y _test=train_test_split(X,Y,test_size=0.2,random_state=42)

scaler_X-StandardScaler()
scaler_Y-StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.fit_transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.fit_transform(Y_test)

sgd-S6DRegressor(=ax_iter-1000,tol-1e-3)
multi_output_sgd-MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)

Y_pred-multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y test=scaler_Y.inverse transform(Y_test)
mse-mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)

print("InPredictions\n",Y_pred[:5])
```


## Output:

![Image](https://github.com/user-attachments/assets/e6a85d91-6c1e-497d-b2fa-bfe3d7d2c562)

![Image](https://github.com/user-attachments/assets/32f0fc47-a847-4605-9116-6946f54d2f96)

![Image](https://github.com/user-attachments/assets/4dfa6a7a-ee49-414e-99ba-a9f4b3004107)

![Image](https://github.com/user-attachments/assets/dca0ee26-db97-4aea-a94c-b92b4c591d45)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
