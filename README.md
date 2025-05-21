# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Keziah.F
RegisterNumber:  212223040094
*/
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
## ENCODING
![image](https://github.com/user-attachments/assets/d3e2f384-d6f3-4a6d-a133-1aa63ca1a0c3)

## Head():

![image](https://github.com/user-attachments/assets/a704d324-ac39-4cd7-a035-81a562b7d785)

## Info():

![image](https://github.com/user-attachments/assets/16b297fc-9acc-47ac-8ec0-b08fd071857f)

## isnul().sum():

![image](https://github.com/user-attachments/assets/352a0781-826d-4cf3-840d-e796c0da0685)

## Prediction of Y

![image](https://github.com/user-attachments/assets/c5d1c084-23f3-4a63-9a25-da2425f702fe)

## Acuuarcy

![image](https://github.com/user-attachments/assets/799a522e-1ead-4353-b879-f1a257de489b)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
