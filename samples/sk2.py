
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
print(type(iris.data), type(iris.target))

# let's do some EDA
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())

# now let's do k neighbors classifier
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
# metric_params=None, n_jobs=1, n_neighbors=6, p=2,  weights='uniform')

knn = KNeighborsClassifier(n_neighbors=6)

# pass features as numpy array and labels to the target also as numpy array
knn.fit(iris['data'], iris['target'])
print(knn)

# Now predict

X_new = np.random.uniform(low=1.0, high=9.0, size= iris['data'].shape)
print("X_new = ", str(X_new))
#print(iris['data'])
prediction = knn.predict(X_new)
print(prediction)

# note in pycharm you need to do this
#plt.show()
