
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
#%matplotlib inline

# wth is this?
plt.style.use('ggplot')

iris = datasets.load_iris()
print(iris.target_names)
print(iris.keys())
# see the type is numpy array..yoo hoo!
print("iris data = ", str(type(iris.data)),  type(iris.target))

# let's do some EDA
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

#X, y = datasets.load_iris(return_X_y=True)
#print(type(X))
#print(type(y))
#exit(0)

# now let's do visual EDA
_ = pd.plotting.scatter_matrix(df, c=iris.target, figsize=(8,8), s=200, marker='G')
plt.show()

# let's do titanic using count plot
sns.set(style="darkgrid")
titanic = sns.load_dataset("titanic")
ax = sns.countplot(x="class", data=titanic)

# let's do tips (wth is tips?_
tips = sns.load_dataset("tips")
sns.catplot(x="day", y="total_bill", data=tips)

# note in pycharm you need to do this
plt.show()