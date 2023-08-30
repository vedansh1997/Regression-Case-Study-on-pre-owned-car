# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:20:30 2023

@author: lenovo
"""

import pandas as pd
import numpy as np
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
car_data= pd.read_csv('cars_sampled.csv')
car_data1=car_data.copy(deep=True)
car_data.info()
car_data.isnull().sum()
description=car_data1.describe()
car_data1.describe()
pd.set_option('display.float_format',lambda x: '%.3f' %x)
car_data1.describe()
pd.set_option('display.max_columns',500)
car_data1.describe()
col = ['name','dateCrawled','dateCreated','postalCode','lastSeen']
car_data1=car_data1.drop(columns=col,axis=1)
car_data1.drop_duplicates(keep='first',inplace=True)


### 
car_data1=car_data1[
           (car_data1.yearOfRegistration<=2018)
           &(car_data1.yearOfRegistration>=1950)
           &(car_data1.price>=100)
           &(car_data1.price<=150000)
           &(car_data1.powerPS>=10)
           &(car_data1.powerPS<=500)]
''' above what we have done is we select the data which these attributes common'''

''' so if you see the data there is no column for the age so we have to crate one
for this we subtract the year Of Registration with 2018 and add the month of 
registration/12 to convert into years'''
car_data1['monthOfRegistration']=car_data1['monthOfRegistration']/12 # dont used this
car_data1.columns
car_data1['monthOfRegistration']/=12 # convert months into years
### CREATING a new variable age by 2018-year of registration +month of registration/12
car_data1['Age']=(2018-car_data1['yearOfRegistration'])+car_data1['monthOfRegistration']
car_data1['Age']=round(car_data1['Age'],2)
car_data1['Age'].describe() # there is very less difference between mean and median\
''' so now we can remove yearOfRegistration and monthOfRegistration it creates
 a redundency'''
car_data1=car_data1.drop(columns=['yearOfRegistration','monthOfRegistration']\
                         ,axis=1)
# remove insignificant variable
col1=['abtest','seller','offerType']
car_data1=car_data1.drop(col1,axis=1)
car_data2=car_data1.copy()

### correlation
car_select=car_data1.select_dtypes(exclude=[object])
correlation=car_select.corr()
correlation=round(correlation,3)
car_select.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

''' Now we are going to build a linear regression or random forest model
on two set of the data
1 Data obtained by omitting row with missing data
2 data obtained by imputing the missing values
'''
 
### Omitting missing values
car_omit=car_data1.dropna(axis=0)

# converting categorical data into dummies
car_omit=pd.get_dummies(data=car_omit,drop_first=True)

'''import important libraries'''
from sklearn.model_selection import train_test_split # used to split data as the
# training and test data
from sklearn.linear_model import LinearRegression # 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# seperating input and output features
x= car_omit.drop(['price'],axis=1,inplace=False) # inplace= false because we don't
# need any change car
y=car_omit['price']

# plotting price
'''but before do the plotting we convert the price into a data frame as in variable 
explorer it shows y as the series'''
prices= pd.DataFrame({'before':y,'after':np.log(y)}) # here i create a data frame
# in wich i create two columns one is the before which contain price, and after 
# contain log value of price
prices.hist()
''' why i used log values  will be clear when i create a histogram as you can see
we get a bell shaped histogram when we used natural logarithm'''
# so now transform the natural logrithmic values 
y=np.log(y)

# spitting the data into test and train data
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=.7,random_state=3)
''' random state is a predefined algorithm and called pseudo random generator 
you can give any value to this parameter but we are giving the value 3 so every 
time you run this algorithm the same set of record go to train and test'''
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

### baseline model for ommitted data
''' we are going to make a base model by using the test data mean value
This set to a benchmark and to compare the regression model'''
## finding the mean of test data value
base_pred=np.mean(y_test)
print(base_pred)
# in a baseline model the predicted value basically replaced by the mean value
# of the testdata and this acchually set the benchmark for us and to compare our 
# model that we are going to made in future

# repeat the same value to till the length of the test data
base_pred=np.repeat(base_pred,len(y_test))

## RmSe (RootmeanSquareError)
rmse_base=np.sqrt(mean_squared_error(y_test, base_pred))
print(rmse_base)
'''so this is the benchmark for the comparision any other model in future will
give you the value of rmse that less than this so that is the objective for us
'''

### linear regression for ommitted data
Lgr=LinearRegression(fit_intercept=True)
# model
linear_model=Lgr.fit(x_train,y_train)
# prdiction
linear_predict=Lgr.predict(x_test)
# comput mse and rmse
orignal_mes=mean_squared_error(y_test, linear_predict)
Rmse= np.sqrt(orignal_mes)
print(Rmse)
# r squared value
r2_train=linear_model.score(x_train,y_train)
r2_test=linear_model.score(x_test,y_test)
print(r2_test,r2_train)

# regression diagnostic: Reasidual Plot Analysis
residual= y_test-linear_predict
sns.regplot(x=linear_predict,y=residual,fit_reg=False,scatter=True)
residual.describe()
# its shows that your mean close to zero and indicate it is a good model
 
'''Random forest with omitted data'''
 ### model Parameter
rf= RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,
                          min_samples_split=10,min_samples_leaf=4,random_state=1)
''' n_estimator told you no of trees in your forest (default 10)
   max_depth is the dept of each tree how much you want each tree goes.
   min_sample_split is the minimum no of sample that is required for a node to
   split ,let say you have two observation in a node and you don't want a node
   any further because it mightnot make any sense so in that case if you want to
   impose scuh condition use this.
   min_samples_leaf it is the no of the sample which is required to be at a leaf
   node
    max_features is the features which basically algorithm want to consider to
    build the model or to build the tree '''
# model
rf_model = rf.fit(x_train,y_train)
# prediction
rf_predict=rf.predict(x_test)

# compute rmse and mse
rf_mse= mean_squared_error(y_test, rf_predict )
rf_rmse=np.sqrt(rf_mse)
print(rf_rmse)
# r squared value
r2_rf_train=rf_model.score(x_train, y_train)
r2_rf_test=rf_model.score(x_test, y_test)
print(r2_rf_test,r2_rf_train)

# model building wih imputing data
car_imputed=car_data1.apply(lambda x:x.fillna(x.median())
                            if x.dtype=='float' else\
                                x.fillna(x.value_counts().index[0]))
car_imputed.isnull().sum()
# converted categorical data into dummies
car_imputed=pd.get_dummies(data=car_imputed,drop_first=True)
# seperating input and output features
x2=car_imputed.drop(['price'],axis='columns',inplace=False)
y2=car_imputed['price']
prices1=pd.DataFrame({'before':y2,'after':np.log(y2)})
prices1.hist()
## Transforming price as a logarithmic value
y2=np.log(y2)
# setting data as the train and test
x_train1,x_test1,y_train1,y_test1=train_test_split(x2,y2,test_size=.3,random_state=3)
print(x_train1.shape,x_test1.shape,y_train1.shape,y_test1.shape)
''' Basemodel for imputed data'''
base1=np.mean(y_test1)
base1=np.repeat(base1,len(y_test1))
rmse1=np.sqrt(mean_squared_error(y_test1, base1))
print(rmse1)
''' Linear Regression with imputed data'''

'''setting intercept as true'''
Lgr1=LinearRegression(fit_intercept=True)

# model
linear_model1=Lgr1.fit(x_train1,y_train1)
# prdiction
linear_predict1=Lgr1.predict(x_test1)
# comput mse and rmse
orignal_mes=mean_squared_error(y_test1, linear_predict1)
rmse_linear1=np.sqrt(orignal_mes)
print(rmse_linear1)
# r squared value
r2_train1=linear_model1.score(x_train1,y_train1)
r2_test1=linear_model1.score(x_test1,y_test1)
print(r2_test1,r2_train1)

'''Random Forest for the imputed data'''
rf2= RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,\
                          min_samples_split=10,min_samples_leaf=4,random_state=1)
# MODEL
model_forest2=rf.fit(x_train1, y_train1)
# test the data
test_rf2=rf.predict(x_test1)
# computing mse and rmse 
mse_rf1=mean_squared_error(y_test1, test_rf2)
rmse_rf1=np.sqrt(mse_rf1)
print(rmse_rf1)
# r-squared
r2_training_rf1=model_forest2.score(x_train1,y_train1)
r2_test_rf1=model_forest2.score(x_test1, y_test1)
print(r2_test_rf1,r2_training_rf1)
