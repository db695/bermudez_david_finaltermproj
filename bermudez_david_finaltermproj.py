#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install scikit-learn


# In[ ]:


pip install tabulate


# In[ ]:


pip install tensorflow


# In[1]:


import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier #DecisionTree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold #K Fold
from tabulate import tabulate #Format OUTPUT
import numpy as np
#RandomForest
from sklearn.ensemble import RandomForestClassifier
# LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical


# In[2]:


file_name = "dataTemplate/diabetes.csv"


# In[3]:


df = pd.read_csv(file_name, encoding='ISO-8859-1',delimiter=',', on_bad_lines='skip')


# In[4]:


df.head()


# In[6]:


X = df.copy()
y = df['Outcome'] #update here
kf = KFold(n_splits=10, shuffle=True, random_state=42) #Splits 10 times
metrics_list = []

headers = [
    "TP","TN","FP", "FN", "TPR", "TNR", "FPR", "FNR", "Precision", "F1 Measure", "Accuracy", "Error Rate",
    "BACC", "TSS", "HSS"
]

def calc_metrics(cm):
    TP, FN = cm[0][0], cm[0][1]
    FP, TN = cm[1][0], cm[1][1]
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (TN + FP)
    FNR = FN / (TP + FN)
    Precision = TP / (TP + FP)
    F1_measure = 2 * TP / (2 * TP + FP + FN)
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    Error_rate = (FP + FN) / (TP + FP + FN + TN)
    BACC = (TPR + TNR) / 2
    TSS = TPR - FPR
    HSS = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
    metrics = [TP, TN, FP, FN, TPR, TNR, FPR, FNR, Precision, F1_measure,Accuracy, Error_rate, BACC, TSS, HSS]
    metrics = [float(m) for m in metrics]
    return metrics


#Decision Tree Below
for i, (train_index, test_index) in enumerate(kf.split(X), start=1):
        # Splitting the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

       #then use the data to feed to the models.
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        result = clf.predict(X_test) #from here calculate the performance values
        cm = confusion_matrix(y_test, result) #print(calc_metrics(cm)) 
        #print(result)
        metrics_list.append(calc_metrics(cm)) #The 10 splits of data with the
print("**")
print("**")
print("Decision Tree")
#use Tabulate, print outside the loop
print(tabulate(metrics_list, headers=headers, tablefmt="grid", floatfmt=".3f"))

#Random Forest Below
rf_metrics_list = []

for i, (train_index, test_index) in enumerate(kf.split(X), start=1):
        # Splitting the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
       #then use the data to feed to the models.
        rf_clf = RandomForestClassifier() #number of trees in the forest = 100 (default) n_estimators=100, min_samples_split=10
        rf_clf = rf_clf.fit(X_train, y_train)
        rf_result = rf_clf.predict(X_test) #from here calculate the performance values
        rf_cm = confusion_matrix(y_test, rf_result) #print(calc_metrics(cm)) 
        rf_metrics = calc_metrics(rf_cm)
        rf_metrics_list.append(rf_metrics)
        #metrics_list.append(calc_metrics(cm)) #The 10 splits of data with the
print("**")
print("**")
print("Random Forest")
print(tabulate(rf_metrics_list, headers=headers, tablefmt="grid", floatfmt=".3f"))
print("**")
print("**")

#LSTM Below
print("LSTM")
X_lstm = np.expand_dims(X.values, axis=1)
y_lstm = to_categorical(y)

model = Sequential()
model.add(LSTM(50, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

LSTM_metrics_list = []

for train_index, test_index in kf.split(X_lstm):
    X_train, X_test = X_lstm[train_index], X_lstm[test_index]
    y_train, y_test = y_lstm[train_index], y_lstm[test_index]

    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test_classes, y_pred_classes)

    metrics = calc_metrics(cm)
    LSTM_metrics_list.append(metrics)

print(tabulate(LSTM_metrics_list, headers=headers, tablefmt="grid", floatfmt=".3f"))   


# In[ ]:




