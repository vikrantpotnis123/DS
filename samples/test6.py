import pandas as pd
import matplotlib.pyplot as plt

# plt.style.use('ggplot')

# read csv using panda
df = pd.read_csv('/Users/vpotnis/Desktop/nba.csv')

print(df[(df['Salary'] >= 3000000.00) & (df['Salary'] < 5000000.00)])
print(df.corr(method='pearson'))
print(df.corr(method='kendall'))

df1 = df['Salary'][(df['Salary'] >= 3000000.00) & (df['Salary'] < 5000000.00)]
df2 = df['Age'][(df['Salary'] >= 3000000.00) & (df['Salary'] < 5000000.00)]

fig, axes = plt.subplots(1, 2)
axes[0].set_xlabel('Salary')
axes[1].set_xlabel('Age')
axes[0].set_title('Salary Histogram')
axes[1].set_title('Age Histogram')
df1.plot(kind='hist', ax=axes[0])
df2.plot(kind='hist', ax=axes[1])

plt.show()
