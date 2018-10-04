'''
Script pour la competition
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

'''
# TODO some cat vars have no values only in the test dataset


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def count_nan(columns):
    for col in columns:
        print("{}: \t{} {}".format(col,
            df[df[col].isnull()].shape[0],
            vdf[vdf[col].isnull()].shape[0]
            ))


# metric de scoring
def log_rmse(yhat, ytrue):
    return np.sqrt(mean_squared_error ( np.log(yhat), np.log(ytrue) ))

if __main__ == '__name__':

    df = pd.read_csv('ames/train.csv')
    vdf = pd.read_csv('ames/test.csv')

    # columns types
    int_columns = [col for col in df.columns if df[col].dtype == 'int64']
    # enlever SaleType
    int_columns = int_columns[:-1]
    # pour traitement ultérieur des variables d'annees et de mois
    cat_int_columns = ['GarageYrBlt', 'YrSold','MoSold', 'YearBuilt']
    int_columns = [col for col in int_columns if col not in cat_int_columns]
    # variables float
    flt_columns = [col for col in df.columns if df[col].dtype == 'float64']
    # variables categorielles
    cat_columns = [col for col in df.columns if df[col].dtype == 'O']

    # -------------------------------------------------------
    #  Missing values
    # -------------------------------------------------------

    # missing values for categories
    # remplacer toutes les valeurs manquantes par 'Other'
    # count_nan(cat_columns)
    for col in cat_columns:
        df[col].fillna('Other', inplace = True)
        vdf[col].fillna('Other', inplace = True)


    # count_nan(int_columns)
    # remplacer toutes les valeurs manquantes par 0
    for col in int_columns:
        df[col].fillna(0, inplace = True)
        vdf[col].fillna(0, inplace = True)

    # count_nan(flt_columns)
    # remplacer toutes les valeurs manquantes par la moyenne des autres valeurs
    for col in flt_columns:
        df[col].fillna(np.mean(df[col]), inplace = True)
        vdf[col].fillna(np.mean(vdf[col]), inplace = True)

    # avant d'encoder ou de normaliser
    # j'ai besoin de considerer les datasets de training et de validation
    # en meme temps
    # adf = df + vdf
    adf = pd.concat([df,vdf])
    # normalize floats
    for col in flt_columns:
        df[col] = df[col] / np.max(adf[col])
        vdf[col] = vdf[col] / np.max(adf[col])

    # label encoding
    for col in cat_columns:
        le = LabelEncoder()
        le.fit(adf[col])
        df[col]  = le.transform(df[col])
        vdf[col] = le.transform(vdf[col])

    # X et y
    X = df[cat_columns + flt_columns + int_columns]
    # prendre le log pour de skewer la variable a predire

    y = np.log( df['SalePrice'] +1  )
    # -----------------------------------------------------
    #  Model
    # -----------------------------------------------------

    clf = RandomForestRegressor(n_estimators=1000)
    param_grid = {"max_depth": [3,6, 9, 12],
                  "min_samples_split": [2, 3, 5]}

    # run grid search
    gs = GridSearchCV(clf, param_grid=param_grid, cv=3, verbose = 2)
    gs.fit(X,y)

    print(np.sqrt(gs.best_score_))
    print(gs.best_params_)

    # juste pour verfier que l'on n'overfit pas
    # sur un split test train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    yhat_train = gs.best_estimator_.predict(X_train)
    yhat_test = gs.best_estimator_.predict(X_test)
    train_score = np.sqrt(mean_squared_error(yhat_train, y_train))
    test_score = np.sqrt(mean_squared_error(yhat_test, y_test))
    print("test {:.4f} train {:.4f} ".format(test_score, train_score))

    # -----------------------------------------------------
    # kaggle submission
    # -----------------------------------------------------
    X_valid = vdf[cat_columns + flt_columns + int_columns]
    yhat_valid = gs.best_estimator_.predict(X_valid)
    yhat_valid = np.exp(yhat_valid) -1

    results = pd.DataFrame(columns = ['Id', 'SalePrice'])
    results['Id'] = X_valid.index + 1461
    results['SalePrice'] = yhat_valid
    results.to_csv("submission_02.csv", index = False)
