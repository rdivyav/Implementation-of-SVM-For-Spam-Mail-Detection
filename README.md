# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries for encoding detection, data handling, text processing, machine learning, and evaluation. 
2. Open the dataset file (spam.csv) in binary mode and detect its character encoding using chardet.
3. Load the CSV file into a pandas DataFrame using the detected encoding (e.g., 'windows-1252').
4. Display the dataset using head(), info(), and check for any missing values with isnull().sum().
5. Assign the label column (v1) to variable x and the message text column (v2) to variable y.
6. Split the dataset into training and testing sets using an 80-20 split with train_test_split().
7. Convert the text data into numerical format using CountVectorizer to prepare for model training.
8. Initialize the Support Vector Machine classifier (SVC) and train it on the vectorized training data.
9. Predict the labels of the test set using the trained SVM model.
10. Evaluate the model’s performance by calculating and printing the accuracy score using accuracy_score().

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Divya R V
RegisterNumber: 212223100005 
*/
```
```
import chardet
file="spam.csv"
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding="windows-1252")
data.head()
data.info()
data.describe()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
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
from sklearn.metrics import accuracy_score,classification_report
accuracy=accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![SVM For Spam Mail Detection](sam.png)
![image](https://github.com/user-attachments/assets/c4e5483b-89d8-4e9b-bcb9-58eb33ae7c43)

![image](https://github.com/user-attachments/assets/2f5fb651-3e8e-46a4-a11a-2b4d8bf3bced)

![image](https://github.com/user-attachments/assets/9b50d620-6d0a-4c19-a5f1-8c9572bf6b52)

![image](https://github.com/user-attachments/assets/f495d474-5477-45e1-942d-f91b6c29dd62)

![image](https://github.com/user-attachments/assets/3f8d047c-e1e0-44f6-8a29-28abd5e9f909)

![image](https://github.com/user-attachments/assets/3632058a-f194-428f-b422-31f716137b38)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
