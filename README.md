# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Iris dataset and split it into training and testing data.

2.Train a Logistic Regression (SGD-based) classifier using the training data.

3.Predict the Iris species for the test data.

4.Evaluate the model using accuracy, confusion matrix, and classification repo
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: kiruthika vasanthi M
RegisterNumber:  212225040189
*/
```
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=load_iris()
x,y=data.data,data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression(multi_class='multinomial',solver='lbfgs')
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy:.2f}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

from sklearn.metrics import classification_report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
```

## Output:
![ex 7 ml](https://github.com/user-attachments/assets/210e62c8-10ab-4f48-be85-275b05f11edd)





## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
