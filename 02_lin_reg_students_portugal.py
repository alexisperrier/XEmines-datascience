import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('../data/student-mat-simplified.csv')

lm = smf.ols(formula='G3 ~ age + Medu + Fedu + traveltime + studytime + failures + famrel + freetime + goout + Dalc + Walc + health + absences', data=df).fit()
lm = smf.ols(formula='G3 ~ Medu + traveltime + studytime + failures + goout  + absences', data=df).fit()
lm = smf.ols(formula='G3 ~ G1 + Medu  + failures + goout ', data=df).fit()
