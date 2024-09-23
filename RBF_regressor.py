import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('ENB2012_data.csv')	
	
corr = df.corr()
plt.figure(figsize=(10, 5))
sns.heatmap(corr, annot=True, fmt='.2f', linewidths=0.5, linecolor='white',  cmap='Blues')
plt.show()

columns_to_drop = ['X6','Y2']
X = df.drop(columns=columns_to_drop, axis=1)
y = df['Y1']  
X	

scaler = MinMaxScaler(feature_range=(0, 1))
df = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
df.describe().loc[['min','mean','std','max']].T.style.background_gradient(axis=1)

trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.1)

sc=StandardScaler()
scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)

kernel = 1.0 * RBF(length_scale=1.0)
rbf_regressor = GaussianProcessRegressor(kernel=kernel, random_state=42)
rbf_regressor.fit(testX_scaled,testY)

y_pred = rbf_regressor.predict(testX_scaled)
df_temp = pd.DataFrame({'Actual': testY, 'Predicted': y_pred})
df_temp.head(30)

df_temp = df_temp.head(30)
df_temp.plot(kind='bar',figsize=(10,6))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(testY, y_pred, label='Actual vs Predicted', color='blue', alpha=0.5)

# Add regression line
plt.plot(testY, testY, color='red', label='Regression Line')

# Add labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Actual vs Predicted Values')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()

import sklearn.metrics as sm

# Calculate regression performance metrics
print("Mean absolute error =", round(sm.mean_absolute_error(trainY, y_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(trainY, y_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(trainY, y_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(trainY, y_pred), 2))
print("R2 score =",(sm.r2_score(trainY, y_pred))*100)