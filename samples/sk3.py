
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import  StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# decision tree
from sklearn.tree import DecisionTreeClassifier
from  sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# wth is this?
plt.style.use('ggplot')

bc = datasets.load_breast_cancer()
print(bc.keys())
print(bc.feature_names)
# see the type is numpy array..yoo hoo!
print(type(bc.data), type(bc.target))

# let's do some EDA
df = pd.DataFrame(bc.data, columns=bc.feature_names)

#print(df.head())
#print(df.shape)
#print("type = ", str(type(df)))

# now let's do visual EDA
#_ = pd.plotting.scatter_matrix(df, c=bc.target, figsize=(10, 10), s=200, marker='G')
#plt.show()


# split dataset into 80% train and 20 % test
X = bc['data']
y = bc['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# fit dt to the training set
#dt.fit(bc['data'], bc['target'])
#print("bc[target]")
#print(bc['target'])
dt.fit(X_train, y_train)
#print(y_train)

# predict test set labels
y_pred = dt.predict(X_test)
#print("y_pred")
#print(y_pred)

# eval accuracy of test set
a = accuracy_score(y_test, y_pred)
print(a)

