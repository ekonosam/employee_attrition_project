import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('notebook/data/HR_Analytics.csv')

df_model = df[['MonthlyIncome', 'YearsAtCompany', 'JobRole', 'Education', 'EducationField']].copy()

# Log transform
df_model['log_MonthlyIncome'] = np.log1p(df_model['MonthlyIncome'])
df_model['log_YearsAtCompany'] = np.log1p(df_model['YearsAtCompany'])

# One-hot encode categorical variables
categorical_cols = ['JobRole', 'Education', 'EducationField']
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded_cols = ohe.fit_transform(df_model[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(categorical_cols))

# Combine dataframes
df_model = pd.concat([df_model.drop(columns=categorical_cols + ['MonthlyIncome']), encoded_df], axis=1)

X = df_model.drop('log_MonthlyIncome', axis=1)
y = df_model['log_MonthlyIncome']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())
