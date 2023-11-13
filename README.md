# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Palleri Yogi
RegisterNumber: 212220040108
*/

import chardet
file = '/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd 
data = pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
Result Output:

![image](https://github.com/YogiReddy117/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123739437/ed59ced0-200f-4896-a3df-120ab8192339)

data.head():

![image](https://github.com/YogiReddy117/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123739437/1fdccef8-fc7d-4683-b98d-ef039f193ebe)

data.info():

![image](https://github.com/YogiReddy117/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123739437/f36af0f9-e80d-4a9f-9a1b-1312fbe3fd39)

data.isnull().sum():

![image](https://github.com/YogiReddy117/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123739437/aa59a2a9-26a5-462f-91e0-fdde1f261e88)

Y_prediction value:

![image](https://github.com/YogiReddy117/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123739437/ad61a80a-8849-4d17-b4e0-97114c68de09)

Accuracy value:

![image](https://github.com/YogiReddy117/Implementation-of-SVM-For-Spam-Mail-Detection/assets/123739437/a907a038-ef50-4975-bc25-c102b173722c)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
