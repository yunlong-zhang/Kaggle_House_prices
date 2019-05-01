import pandas as pd
import numpy as np
import tensorflow as tf
import datetime as dt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

### Print a head fore better presentation in Terminal.
print('\n' * 5)
print('='*80)
print(dt.datetime.now())
print(tf.VERSION)
print(tf.keras.__version__)
print('='*80)
print('\n')

### Import the values
train = pd.read_csv('/Volumes/GoogleDrive/My Drive/Python/Kaggle/House_prices/train.csv')
exam = pd.read_csv('/Volumes/GoogleDrive/My Drive/Python/Kaggle/House_prices/test.csv')
train = train.set_index('Id')
exam = exam.set_index('Id')
# print(train.head())
# print(train.shape)

### Remove the features with more than 100 null values.
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
null_features = list(nulls[nulls.iloc[:,0] > 100].index)
train = train.drop(null_features, axis=1)
exam = exam.drop(null_features, axis=1)
# print(train.columns)

### Plot the values to select reprocessing methods.
def check_non_num(df):
	print(df.shape)
	non_numeric = df.select_dtypes(exclude=[np.number])
	for i in non_numeric.columns:
		print('====' + i)
		df = df.fillna('TestNA')
		quality_pivot = df.pivot_table(index=i, values='SalePrice', aggfunc=np.median)
		print(df[i].value_counts())
		print('Total:', df[i].value_counts().sum())
		quality_pivot.plot(kind='bar')
		plt.xlabel(i)
		plt.ylabel('Sales Price')
		plt.show()

def check_num(df):
	print(df.shape)
	numeric = df.select_dtypes(include=np.number)
	not_categorical = []
	for i in numeric.columns:
		print('====' + i)
		if len(df[i].unique()) > 10:
			print('Too many values, not a categorical feature.')
			not_categorical.append(i)
		else:
			quality_pivot = df.pivot_table(index=i, values='SalePrice', aggfunc=np.median)
			print(df[i].value_counts())
			print('Total:', df[i].value_counts().sum())
			quality_pivot.plot(kind='bar')
			plt.xlabel(i)
			plt.ylabel('Sales Price')
			plt.show()
	print('Features not categorical:', not_categorical)

### By the previous section, drop the non_numeric features have no enough instance for learning.
non_num_not_corr = ['SaleType', 'GarageCond', 'GarageType', 'Exterior2nd', 'RoofStyle', 'BldgType', 'GarageQual', 'Functional', 'Electrical', 'CentralAir', 'Heating', 'BsmtFinType2', 'BsmtCond', 'ExterCond', 'RoofMatl', 'Condition2', 'Condition1', 'LandSlope', 'LotConfig', 'Utilities', 'LandContour', 'Street', 'MSZoning']
train = train.drop(non_num_not_corr, axis=1)
exam = exam.drop(non_num_not_corr, axis=1)
# check_non_num(train)
train_num = train.select_dtypes(include = np.number)
# check_num(train_num)
corr = train_num.corr().SalePrice[:-1]
corr_columns = corr[corr.values ** 2 >= 0.16].index.tolist()
not_corr_columns = corr[corr.values ** 2 <= 0.16].index.tolist()
# print('Features correlated:', corr_columns)
# print('Features not correlated:', not_corr_columns)
# check_num(train.loc[:, not_corr_columns + ['SalePrice']])

selected_features = corr_columns
###
# Encoding numeric columns that are categorical and not correlated with SalePrice.
###

### No good features for this encoding!!!

###
# Encoding non-numeric columns
###

### Set LotShape to Reg 0, Other 1.
def enc_LotShape(x): return 0 if x=='Reg' else 1
train['LotShape'] = train.LotShape.apply(enc_LotShape)
exam['LotShape'] = exam.LotShape.apply(enc_LotShape)
selected_features.append('LotShape')

### Set Neighborhood in 3 bins.
pivot = train.pivot_table(index='Neighborhood', values='SalePrice', aggfunc=np.median)
new_labels = pd.cut(pivot.SalePrice, 3, labels=[0, 1, 2]).to_dict()
train.replace({'Neighborhood': new_labels}, inplace=True)
exam.replace({'Neighborhood': new_labels}, inplace=True)
selected_features.append('Neighborhood')

### HouseStyle. 2.5Fin and 2Story to 2, SLvl and 1Story to 1, others to 0.
def enc_HouseStyle(x):
	if x=='2.5Fin' or x == '2Story':
		return 2
	elif x == 'SLvl' or '1Story':
		return 1
	else:
		return 0
train['HouseStyle'] = train.HouseStyle.apply(enc_HouseStyle)
exam['HouseStyle'] = exam.HouseStyle.apply(enc_HouseStyle)
selected_features.append('HouseStyle')

### Set Exterior1st in 3 bins.
pivot = train.pivot_table(index='Exterior1st', values='SalePrice', aggfunc=np.median)
new_labels = pd.cut(pivot.SalePrice, 3, labels=[0, 1, 2]).to_dict()
train.replace({'Exterior1st': new_labels}, inplace=True)
exam.replace({'Exterior1st': new_labels}, inplace=True)
selected_features.append('Exterior1st')

### MasVnrType. Stone 2, BrkFace 1 and others 0. Fillna 1.
def enc_MasVnrType(x):
	if x=='Stone':
		return 2
	elif x == 'BrkFace':
		return 1
	else:
		return 0
train['MasVnrType'] = train.MasVnrType.fillna(value = 'BrkFace')
train['MasVnrType'] = train.MasVnrType.apply(enc_MasVnrType)
exam['MasVnrType'] = exam.MasVnrType.fillna(value = 'BrkFace')
exam['MasVnrType'] = exam.MasVnrType.apply(enc_MasVnrType)
selected_features.append('MasVnrType')

### Foundation. Others 0, PConc 1.
def enc_Foundation(x): return 1 if x=='PConc' else 0
train['Foundation'] = train.Foundation.apply(enc_Foundation)
exam['Foundation'] = exam.Foundation.apply(enc_Foundation)
selected_features.append('Foundation')

### ExterQual and KitchenQual. Convert [Ex, Gd, TA, Fa] to [3,2,1,0]
new_labels = {'Ex': 3, 'Gd': 2, 'TA': 1, 'Fa': 0}
train.replace({'ExterQual': new_labels}, inplace=True)
train.replace({'KitchenQual': new_labels}, inplace=True)
exam.replace({'ExterQual': new_labels}, inplace=True)
exam.replace({'KitchenQual': new_labels}, inplace=True)
selected_features.append('KitchenQual')

### BsmtQual. [Ex, Gd, TA, Fa] to [3, 2, 1, 0]. Fillna 0.
train.replace({'BsmtQual': new_labels}, inplace=True)
train.BsmtQual = train.BsmtQual.fillna(0)
exam.replace({'BsmtQual': new_labels}, inplace=True)
exam.BsmtQual = exam.BsmtQual.fillna(0)
selected_features.append('BsmtQual')

### BsmtExposure. [Gd, Av, Mn, No, null] to [4,3,2,1,0].
new_labels = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, np.nan:0}
train.replace({'BsmtExposure': new_labels}, inplace=True)
exam.replace({'BsmtExposure': new_labels}, inplace=True)
selected_features.append('BsmtExposure')

### BsmtFinType1. to null 0, others 1, GLQ 2.
new_labels = {'GLQ': 2, np.nan: 0}
train.replace({'BsmtFinType1': new_labels}, inplace=True)
exam.replace({'BsmtFinType1': new_labels}, inplace=True)
def enc_BsmtFinType1(x):
	if x not in [0, 2]:
		return 1
	else:
		return x
train.BsmtFinType1 = train.BsmtFinType1.apply(enc_BsmtFinType1)
exam.BsmtFinType1 = exam.BsmtFinType1.apply(enc_BsmtFinType1)
selected_features.append('BsmtFinType1')

### HeatingQC. Convert [Ex, Gd, TA, Fa, Po] to [4, 3, 2, 1, 0]
new_labels = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}
train.replace({'HeatingQC': new_labels}, inplace=True)
exam.replace({'HeatingQC': new_labels}, inplace=True)
selected_features.append('HeatingQC')

### GarageFinish. Convert [Fin, RFn, Unf, null] to [3, 2, 1, 0]
new_labels = {'Fin': 3, 'RFn': 2, 'Unf': 1, np.nan: 0}
train.replace({'GarageFinish': new_labels}, inplace=True)
exam.replace({'GarageFinish': new_labels}, inplace=True)
selected_features.append('GarageFinish')

### PavedDrive. Convert [Y, P, N] to [2, 1, 0]
new_labels = {'Y': 2, 'P': 1, 'N': 0}
train.replace({'PavedDrive': new_labels}, inplace=True)
exam.replace({'PavedDrive': new_labels}, inplace=True)
selected_features.append('PavedDrive')

### SaleCondition. Others 0. Partial 1.
def enc_SaleCondition(x): return 1 if x == 'Partial' else 0
train.SaleCondition = train.SaleCondition.apply(enc_SaleCondition)
exam.SaleCondition = exam.SaleCondition.apply(enc_SaleCondition)
selected_features.append('PavedDrive')

# print(train[selected_features].isnull().sum())
# Fill np.nan with specific values of GarageYrBlt.
test_pivot = train.pivot_table(index='GarageYrBlt', values='SalePrice')
mean_over_year = test_pivot.describe().iloc[1,0]
train['GarageYrBlt'].fillna(mean_over_year, inplace=True) # Note may not be right!
exam['GarageYrBlt'].fillna(mean_over_year, inplace=True)
# Fill np.nan with specific values of MasVnrArea.
train['MasVnrArea'].fillna(0, inplace=True)
exam.fillna(0, inplace=True)
# print(exam[selected_features].isnull().sum())

###
# Training
###

# Train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[selected_features], np.log(train['SalePrice']), test_size=0.33)
# Linear Regression traing.
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print('LinearRegression R^2 is:', model.score(X_test, y_test))
predictions = model.predict(X_test)
print('LinearRegression MSE is:', mean_squared_error(y_test, predictions))
# Ridge Regression training.
for i in range(-3, 4):
	alpha = 10 ** i
	rm = linear_model.Ridge(alpha=alpha)
	ridge_model = rm.fit(X_train, y_train)
	pred_ridge = ridge_model.predict(X_test)
	
# 	plt.scatter(pred_ridge, y_test, alpha=0.5, color='r')
# 	plt.xlabel('Predicted values (ridge regression)')
# 	plt.ylabel('Actual values')
# 	plt.title('Ridge regularization with alpha = {}'.format(alpha))
# 	overlay = "R^2 is {}\nMSE is: {}".format(
# 			ridge_model.score(X_test, y_test),
# 			mean_squared_error(pred_ridge, y_test))
# 	plt.annotate(s=overlay, xy=(12.1, 10.6), size='x-large')
# 	plt.show()

exam['SalePrice'] = np.e ** lr.predict(exam[selected_features])
submission = exam['SalePrice']
submission.to_csv('result.csv', header=['SalePrice'])
	
