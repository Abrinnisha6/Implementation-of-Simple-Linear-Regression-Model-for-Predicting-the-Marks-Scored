# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

### STEP 1:  Import the needed packages

### STEP 2: Assigning hours To X and Scores to Y

### STEP 3 :Plot the scatter plot

### STEP 4 :Use mse,rmse,mae formmula to find 

## Program:

Program to implement the simple linear regression model for predicting the marks scored.
### Developed by: ABRIN NISHA A
### RegisterNumber: 212222230005

### df.head() 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
#displaying the content in datafile
df.head()
```

### df.tail()
```
df.tail()
```

### Array value of X 

```
X = df.iloc[:,:-1].values
X
```

### Array value of Y

```
Y = df.iloc[:,1].values
Y
```

### Values of Y prediction 

```
Y_pred
```

### Array values of Y test
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
```

### Training Set Graph
```
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
### Test Set Graph 
```
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="skyblue")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

### Values of MSE,MAE AND RMSE 
```
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```


## Output :

![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
