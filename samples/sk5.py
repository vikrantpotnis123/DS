from sklearn import datasets
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
# decision tree
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# wth is this?
plt.style.use('ggplot')

# read csv using panda
df  = pd.read_csv('/Users/vpotnis/Desktop/auto-mpg.csv',  error_bad_lines=False)
print(df.keys())
df['split'] = np.random.randn(df.shape[0], 1)
msk = np.random.rand(len(df)) <= 0.8
#X = df[msk].drop('%mpg', axis=1)
#y = df[~msk].drop('%mpg', axis=1)
X = df.drop(['%mpg', 'model year', 'origin', 'car name'], axis=1)
y = df['%mpg']

#exit(0)

# now let's do visual EDA
#_ = pd.plotting.scatter_matrix(df, c=df["%mpg"], figsize=(10, 10), s=200, marker='G')
#plt.show()

# split dataset into 80% train and 20 % test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# instantiate dt
# dt = DecisionTreeClassifier(max_depth=2, random_state=1)
dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)

# fit dt to the training set
dt.fit(X_train, y_train)
# print(y_train)

# predict test set labels
y_pred = dt.predict(X_test)

# eval root mse
mse_dt = MSE(y_test, y_pred)
rmse_dt = mse_dt ** (1/2)
print(rmse_dt)
