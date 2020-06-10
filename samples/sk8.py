import pandas as pd

df = pd.read_csv('http://bit.ly/kaggletrain', nrows=6)
print(df.columns)
cols = ['Fare', 'Embarked', 'Sex', 'Age']
X = df[cols]
print(X)

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

ohe = OneHotEncoder()
imp = SimpleImputer()

ct = make_column_transformer(
    (ohe, ['Embarked', 'Sex']),  # apply OneHotEncoder to Embarked and Sex
    (imp, ['Age']),  # apply SimpleImputer to Age
    remainder='passthrough')  # include remaining column (Fare) in the output

# column order: Embarked (3 columns), Sex (2 columns), Age (1 column), Fare (1 column)
ct.fit_transform(X)

print(ct)


def test_one_hot_encoder():
    import sklearn.preprocessing as preprocessing
    genders = ['female', 'male']
    locations = ['from Africa', 'from Asia', 'from Europe', 'from US']
    browsers = ['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']
    enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers])
    # Note that for there are missing categorical values for the 2nd and 3rd
    # feature
    X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
    enc.fit(X)
    x = enc.transform([['female', 'from Asia', 'uses Chrome']]).toarray()
    print(x)
    x1 = enc.transform([['female', 'from Europe', 'uses Safari']]).toarray()
    print(x1)
    print(enc.categories_)



test_one_hot_encoder()
