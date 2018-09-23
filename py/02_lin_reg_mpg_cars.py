import pandas as pd
import statsmodels.formula.api as smf

# df = pd.read_csv('../data/student-mat-simplified.csv')
#
# lm = smf.ols(formula='G3 ~ age + Medu + Fedu + traveltime + studytime + failures + famrel + freetime + goout + Dalc + Walc + health + absences', data=df).fit()
# lm = smf.ols(formula='G3 ~ Medu + traveltime + studytime + failures + goout  + absences', data=df).fit()
# lm = smf.ols(formula='G3 ~ G1 + G2 ', data=df).fit()

df = pd.read_csv('../data/autos_mpg.csv')
lm = smf.ols(formula='mpg ~ cylinders + displacement + horsepower + weight + acceleration + origin ', data=df).fit()
lm.summary()
