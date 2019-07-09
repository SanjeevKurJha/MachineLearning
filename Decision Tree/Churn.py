# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:58:23 2017

@author: Sanjeev Jha
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, average_precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz
import warnings
warnings.filterwarnings("ignore")
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"



df=pd.read_csv("Churn.csv",sep=',')

df.shape
df.size
df.head()
df=df.drop(['customerID'],1)
df.tail()

#check for missing value 
pd.isnull(df).any()

pd.isnull(df).sum()

df.info()
df.describe()

# find out thge number whose churn vs none
print("Counts of label")
df.groupby("Churn").size()

#dUMMY CODING CHURN
for col in df.columns:
    if df[col].dtype=='object':
        df[col]=pd.Categorical(df[col]).codes
        
df.head()
df.describe()
df.corr()

df.info()
#X=df.drop(['StreamingMovies','OnlineBackup','Contract','Churn'],1)

X=df.drop(['Churn'],1)

y=df['Churn']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

X_train.shape
X_test.shape
y_train.shape
y_test.shape
X_train.columns

clf1 = tree.DecisionTreeClassifier(random_state=42)
clf2 = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=19, min_samples_split=100,random_state=42)

clf1=clf1.fit(X_train,y_train)
clf2=clf2.fit(X_train,y_train)

y_pred1=clf1.predict(X_test)
y_pred2=clf2.predict(X_test)

print("Model Accuracy1")
metrics.accuracy_score(y_test,y_pred1)
print("Classification Report1")
print(metrics.classification_report(y_test,y_pred1,labels=None, target_names=None, sample_weight=None, digits=2))
print("Roc Auc1")
print(metrics.roc_auc_score(y_test,y_pred1))

print("Model Accuracy2")
metrics.accuracy_score(y_test,y_pred2)
print("Classification Report2")
print(metrics.classification_report(y_test,y_pred2,labels=None, target_names=None, sample_weight=None, digits=2))
print("Roc Auc2")
print(metrics.roc_auc_score(y_test,y_pred2))

#Confusion Matrix
print("Confisusion Matrix1")
cf2=metrics.confusion_matrix(y_test,y_pred1)
lbl1=["Predicted 0","Predicted1"]
lbl2=["True 0","True 1"]
sns.heatmap(cf2,annot=True,cmap="Greens",fmt='d',xticklabels=lbl1,yticklabels=lbl2)
plt.show();


#Confusion Matrix
print("Confisusion Matrix2")
cf2=metrics.confusion_matrix(y_test,y_pred2)
lbl1=["Predicted 0","Predicted1"]
lbl2=["True 0","True 1"]
sns.heatmap(cf2,annot=True,cmap="Greens",fmt='d',xticklabels=lbl1,yticklabels=lbl2)
plt.show();

features1=pd.DataFrame(clf1.feature_importances_,X.columns)
features2=pd.DataFrame(clf2.feature_importances_,X.columns)
features1.columns=["Importance"]
features2.columns=["Importance"]
features1
features2

export_graphviz(clf1,
                out_file="tree_churn1.dot",
                feature_names=X.columns,
                rounded=True,
                filled=True)


export_graphviz(clf2,
                out_file="tree_churn2.dot",
                feature_names=X.columns,
                rounded=True,
                filled=True)


