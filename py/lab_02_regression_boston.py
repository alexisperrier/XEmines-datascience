'''
CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS - proportion of non-retail business acres per town.
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX - nitric oxides concentration (parts per 10 million)
RM - average number of rooms per dwelling
AGE - proportion of owner-occupied units built prior to 1940
DIS - weighted distances to five Boston employment centres
RAD - index of accessibility to radial highways
TAX - full-value property-tax rate per $10,000
PTRATIO - pupil-teacher ratio by town
B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT - % lower status of the population
MEDV - Median value of owner-occupied homes in $000's
'''
import pandas as pd # conventional alias
from sklearn.datasets import load_boston

dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

# correlation
corr  = df.corr(method='pearson')

corr.sort_values(by = 'target', inplace = True)

# find pairs with high correlations

import seaborn as sns # just a conventional alias, don't know why
fig, ax = plt.subplots(figsize=(10, 10))

sns.corrplot(df, ax = ax)

fig, ax = plt.subplots(figsize=(10, 10))
sns.distplot(attr)

import matplotlib.pyplot as plt
attr = df.target
plt.hist(attr)


plt.scatter(df.target, df['LSTAT'])
sns.jointplot(df.target, df['LSTAT'], kind='scatter')
sns.jointplot(df.target, df['LSTAT'], kind='hex')


### explore some diagnostic plots: QQ
## use vif() to compute variance inflation factors

### in the last regression $age has a high p-value
### run regression using all predictors but one: $age
lm.fit1 <- lm(medv ~. -age, data = Boston)
summary(lm.fit1)

summary(lm(medv ~ lstat*age , data = Boston ))

### non-linear transformation of the predictors
lm.fit2 <- lm(medv ~lstat + I(lstat ^2), data = Boston)
summary(lm.fit2)

### now use higher order polynomials with poly()
lm.fit5 = lm(medv ~ poly(lstat, 5), data = Boston)
summary(lm.fit5)

### additional polynomial terms leads to an improvement in the
### model fit

### now try a log transformation for $rm
summary (lm(medv ~ log(rm) , data = Boston ) )


https://www.kaggle.com/sagarnildass/predicting-boston-house-prices
