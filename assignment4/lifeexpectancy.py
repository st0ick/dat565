import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score




#pd.set_option('display.max_columns', None)

df = pd.read_csv("assignment4/life_expectancy.csv")

SEED = 1234
df_train, df_test = train_test_split(df, random_state=1234)

train_corr = df_train.corr(numeric_only=True)['Life Expectancy at Birth, both sexes (years)'].drop('Life Expectancy at Birth, both sexes (years)')
print(train_corr.max(), train_corr.idxmax())


'''
for (columnName, columnData) in df_train.drop('Country', axis=1).items():
    plt.scatter(df_train.drop('Country', axis=1)[columnName], df_train.drop('Country', axis=1)['Life Expectancy at Birth, both sexes (years)'], s=0.4)
    plt.xlabel(columnName)
    plt.ylabel('Life Expectancy at Birth')
    plt.show()
'''
print('\n\n###########\nLinear, one variable regression\n\n')
df_train_linear = df_train[['Human Development Index (value)','Life Expectancy at Birth, both sexes (years)']].dropna()  # Remove the instances with NaN values in HDI or LED 
df_test_linear = df_test[['Human Development Index (value)','Life Expectancy at Birth, both sexes (years)']].dropna()
X_train = np.array(df_train_linear['Human Development Index (value)']).reshape(-1,1)
y_train = np.array(df_train_linear['Life Expectancy at Birth, both sexes (years)'])
X_test = np.array(df_test_linear['Human Development Index (value)']).reshape(-1,1)
y_test = np.array(df_test_linear['Life Expectancy at Birth, both sexes (years)'])

one_var_mdl = LinearRegression().fit(X_train, y_train)
y_train_pred = one_var_mdl.predict(X_train)

slope = one_var_mdl.coef_
intercept = one_var_mdl.intercept_
coef_determ = r2_score(y_train, y_train_pred)

print(f'Slope is: {slope[0]}')
print(f'Intercept is: {intercept}')
print(f'Coefficient of determination: {coef_determ}')

y_test_pred = one_var_mdl.predict(X_test)
pearsonr_test = np.corrcoef(y_test, y_test_pred)[0][1]
mse = mean_squared_error(y_test, y_test_pred)
print(f'Pearson r: {pearsonr_test}')
print(f'Mean Squared Error: {mse}')

plt.scatter(X_train,y_train, s=0.4)
plt.plot(X_train, y_train_pred, c='r')
plt.xlabel('Human Development Index')
plt.ylabel('Life Expectancy at Birth')
#plt.savefig('assignment4/Figure 1.pdf')

plt.show()

print('\n\n###########\n Non-linear regression\n\n')

'''
for (columnName, columnData) in tst.items():
    if (columnName == 'Carbon dioxide emissions per capita (production) (tonnes)'):
        tmp = tst[[columnName, 'Life Expectancy at Birth, both sexes (years)']].dropna()
        tmp_x = tmp[columnName]
        tmp_y = tmp['Life Expectancy at Birth, both sexes (years)']
        print(f'Initial correlation of {columnName} is: {np.corrcoef(tmp_x, tmp_y)}')
        print(f'Log correlation is: {np.corrcoef(np.log(tmp_x), tmp_y)}')
        plt.scatter(tmp_x, tmp_y, s=0.4)
        plt.xlabel(columnName)
        plt.ylabel('assignment4/Life Expectancy at Birth')
        plt.savefig('Figure 2.pdf')
        plt.show()
        plt.scatter(np.log(tmp_x), tmp_y, s=0.4, c='r')
        plt.xlabel('Logarithm of ' + columnName)
        plt.ylabel('assignment4/Life Expectancy at Birth')
        plt.savefig('Figure 3.pdf')
        plt.show()
'''
#necessary to do the following again, since carbon dioxide might have different entries with NaN        
df_train_non_linear = df_train[['Carbon dioxide emissions per capita (production) (tonnes)', 'Life Expectancy at Birth, both sexes (years)']].dropna()
non_linear_x = df_train_non_linear['Carbon dioxide emissions per capita (production) (tonnes)']
non_linear_x_log = np.log(non_linear_x)
non_linear_y = df_train_non_linear['Life Expectancy at Birth, both sexes (years)']

print(f'Initial correlation is: {np.corrcoef(non_linear_x, non_linear_y)[0][1]}')
print(f'Log correlation is: {np.corrcoef(non_linear_x_log, non_linear_y)[0][1]}')

plt.scatter(non_linear_x, non_linear_y, s=0.4)
plt.xlabel('Carbon dioxide emissions per capita (production) (tonnes)')
plt.ylabel('Life Expectancy at Birth, both sexes (years)')
plt.show()

plt.scatter(non_linear_x_log, non_linear_y, s=0.4, c='r')
plt.xlabel('Logarithm of Carbon dioxide emissions per capita (production) (tonnes)')
plt.ylabel('Life Expectancy at Birth, both sexes (years)')
plt.show()

print('\n\n###########\n Multiple linear regression\n\n')

plt.figure(figsize=(27, 10))
max_corr_index = train_corr.drop('Human Development Index (value)').abs()
max_corr_index = max_corr_index[max_corr_index > 0].sort_values(ascending=False).index
#max_corr_index = train_corr.drop('Human Development Index (value)').abs().sort_values(ascending=False).head(10).index
'''
max_corr_matrix = df_train[max_corr_index].corr()
heatmap = sns.heatmap(max_corr_matrix, vmin=-1, vmax=1, annot=True)
plt.savefig('heatmap10.pdf', dpi=300, bbox_inches='tight')

plt.figure(figsize=(16, 6))
chosen_variables = ['Crude Birth Rate (births per 1,000 population)', 'Coefficient of human inequality', 'Rate of Natural Change (per 1,000 population)']
sns.heatmap(df_train[chosen_variables].corr(), vmin=-1, vmax=1, annot=True)
plt.savefig('chosenvar2.pdf', dpi=300, bbox_inches='tight')
'''

max_coef_determ = 0
max_coef_determ_feature = ''
new_max_found = False
features = max_corr_index.tolist()
chosen_features = []
multiple_var_mdl = LinearRegression()
for each in features:
    if max_coef_determ > 0.95:
        break
    for feature in features:

        df_train_multiple = df_train[chosen_features + [feature] + ['Life Expectancy at Birth, both sexes (years)']].dropna()  # Remove the instances with NaN values in HDI or LED 
        df_test_multiple = df_test[chosen_features + [feature] + ['Life Expectancy at Birth, both sexes (years)']].dropna()
        X_train = df_train_multiple[chosen_features + [feature]]
        y_train = df_train_multiple['Life Expectancy at Birth, both sexes (years)']
        #X_test = df_test_multiple[chosen_features + [feature]]
        #y_test = df_test_multiple['Life Expectancy at Birth, both sexes (years)']

        multiple_var_mdl = LinearRegression().fit(X_train, y_train)
        y_train_pred = multiple_var_mdl.predict(X_train)

        coef_determ = r2_score(y_train, y_train_pred)
        if coef_determ > max_coef_determ:
            max_coef_determ = coef_determ
            max_coef_determ_feature = feature
            new_max_found = True
            print(f'Coefficient of determination: {coef_determ}')
    if new_max_found:
        features.remove(max_coef_determ_feature)
        chosen_features.append(max_coef_determ_feature)
        new_max_found = False
    else:
        break

df_train_multiple = df_train[chosen_features + ['Life Expectancy at Birth, both sexes (years)']].dropna()  # Remove the instances with NaN values in HDI or LED 
df_test_multiple = df_test[chosen_features  + ['Life Expectancy at Birth, both sexes (years)']].dropna()
X_train = df_train_multiple[chosen_features]
y_train = df_train_multiple['Life Expectancy at Birth, both sexes (years)']
X_test = df_test_multiple[chosen_features]
y_test = df_test_multiple['Life Expectancy at Birth, both sexes (years)']

multiple_var_mdl = LinearRegression().fit(X_train, y_train)
y_train_pred = multiple_var_mdl.predict(X_train)

slope = multiple_var_mdl.coef_
intercept = multiple_var_mdl.intercept_

print(f'Coefficients are:\n {list(zip(multiple_var_mdl.coef_, chosen_features))}')
print(f'Intercept is: {intercept}')
print(f'Coefficient of determination: {max_coef_determ}')

y_test_pred = multiple_var_mdl.predict(X_test)
pearsonr_test = np.corrcoef(y_test, y_test_pred)[0][1]
mse = mean_squared_error(y_test, y_test_pred)
print(f'Pearson r: {pearsonr_test}')
print(f'Mean Squared Error: {mse}')

print(len(chosen_features))
print(chosen_features)