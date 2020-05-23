import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# read csv using panda
df = pd.read_csv('/Users/vpotnis/Desktop/nba.csv')

print(df[df['Salary'] >= 3000000.00])
print(df['Salary'].value_counts())
print(df.info())

print(df.corr(method='pearson'))
print(df.corr(method='kendall'))

df['Salary'][(df['Salary'] >= 3000000.00) & (df['Salary'] < 5000000.00)].plot(kind='hist')
plt.xlabel('Salary')
plt.ylabel('Freq')
plt.title('Salary Histogram')
plt.show()

df['Age'][(df['Salary'] >= 3000000.00) & (df['Salary'] < 5000000.00)].plot(kind='hist')
plt.xlabel('Salary')
plt.xlabel('Age - Salary')
plt.ylabel('Freq')
plt.title('Age-Salary Histogram')
plt.show()
