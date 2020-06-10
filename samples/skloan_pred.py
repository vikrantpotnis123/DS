import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

mydebug = 1


def make_df(train_file, test_file):
    global train, test, numeric_features, categorical_features
    '''
    train = pd.read_csv(train_file, error_bad_lines=False)
    test = pd.read_csv(test_file, error_bad_lines=False)
    '''

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    train = train.drop('Loan_ID', axis=1)
    numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = train.select_dtypes(include=['object']).drop(['Loan_Status'], axis=1).columns


def split_train_data():
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(train, train['Loan_Status'], test_size=0.2, random_state=1)


def make_column_transformer():
    global numeric_transformer, categorical_transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])


def apply_column_transformer():
    global preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])


def run_random_forest_classifier_best_params():
    param_grid = dict(classifier__n_estimators=[200, 500], classifier__max_features=['auto', 'sqrt', 'log2'],
                      classifier__max_depth=[4, 5, 6, 7, 8], classifier__criterion=['gini', 'entropy'])

    rf = Pipeline(steps=[("preprocessor", preprocessor),
                         ("classifier", RandomForestClassifier())])

    # rf.fit(X_train, y_train)

    # y_pred = rf.predict(X_test)
    # print("RandomForestClassifier", y_pred)

    cv = GridSearchCV(rf, param_grid, n_jobs=1)

    cv.fit(X_train, y_train)
    print("RandomForest classifier Best params " + cv.best_params_)
    print("RandomForest classifier Best score " + cv.best_score_)


def run_classifiers():
    '''

    :return:
    '''
    classifiers_str = ["KNeighborsClassifier", "SVC", "NuSVC",
                       "DecisionTreeClassifier", "RandomForestClassifier",
                       "AdaBoostClassifier", "GradientBoostingClassifier"
                       ]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="rbf", C=0.025, probability=True),
        NuSVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier()
    ]
    l = len(classifiers)
    for classifier, i in zip(classifiers, range(l)):
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
        pipe.fit(X_train, y_train)
        # print(classifier)
        print("%s model score: %.3f percent" % (classifiers_str[i], pipe.score(X_test, y_test) * 100))


def do_main():
    # make data frame
    make_df('/Users/vpotnis/Desktop/cheat_sheets/Loan_prediction/train.csv',
            '/Users/vpotnis/Desktop/cheat_sheets/Loan_prediction/test.csv')

    # print
    if mydebug:
        print(train.keys())
        print(test.keys())

    # split train data into train and test
    split_train_data()

    # print
    if mydebug:
        print(X_train.describe())
        print(X_test.describe())
        print(y_train.describe())
    print(y_test.describe())

    # make col transformers
    make_column_transformer()

    # apply em
    apply_column_transformer()
    if mydebug:
        print(preprocessor)

    # grid search for best params
    # run_random_forest_classifier_best_params()

    # run classifiers
    run_classifiers()




if __name__ == "__main__":
    print("main")
    do_main()
