# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 08:45:13 2023

@author: lenovo
"""

'''Regression Case Study on Predicting Price of Pre owned car '''

import pandas as pd
import numpy as np
import seaborn as sns

## now we are setting the dimension of the plot that i am going to generating
sns.set(rc={'figure.figsize':(11.7,8.27)})
''' here set is a function from seaborn package with the paranthesis i am giving
the figure size '''
car_data=pd.read_csv('cars_sampled.csv')
car_data.isnull().sum()
car_data1=car_data.copy(deep=True)
car_data1.info()  # structure of the data

''' Summarize the data '''
description =car_data1.describe()
# to get rid of ... in the console of the above code result
pd.set_option('display.float_format',lambda x: '%.3f' %x)
# in paranthesis we wrote display.floatformat because the above values are float
# and there lambda function indside and %.3f convert this float into 3 decimal places
car_data1.describe()
## but still we dont able to see the whole data
''' to maximize the no of the column '''
pd.set_option('display.max_columns',500) # 500 means it show max 500 columns in 
#console if there is in the data
car_data1.describe()

''' Dropping unwanted columns '''
col = ['name','dateCrawled','dateCreated','postalCode','lastSeen']
car_data1=car_data1.drop(columns=col,axis=1)
### we can use postal code and carname for analysis but here we use remooved it
''' Remove Dulpicate records '''
car_data1.drop_duplicates(keep='first',inplace=True)
# 470 duplicates
''' Data Cleaning '''
# no of missing values in each column
car_data1.isnull().sum()

### variable year of registration
# now in this we will do year wise countt
yearwise_count=car_data1['yearOfRegistration'].value_counts().sort_index()
# if ypu observe the index it contain the year and 'yearOfRegistration'  contain
# the frequency year wise
sum(car_data1['yearOfRegistration']<1950)
sum(car_data1['yearOfRegistration']>2018)
sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=car_data1)
### working range 1950 to 2018


### variable price
price_count=car_data1['price'].value_counts().sort_index()
sns.distplot(car_data1['price'],kde=False)
car_data1['price'].describe()
sns.boxenplot(y=car_data1['price']) # can't see the box
# so we assume the cost of car between 100$ and 500000$
sum(car_data1['price']>150000)
sum(car_data1['price']<100)
# working range between 100 and 150000$


### variable Power count
power_count=car_data1['powerPS'].value_counts().sort_index()
sns.displot(car_data1['powerPS'])
car_data1['powerPS'].describe()
sns.boxenplot(y=car_data1['powerPS'])
sns.regplot(data=car_data1,x='powerPS',y='price',scatter=True,fit_reg=False)
# so here i am fixing a range between 10 and 500
sum(car_data1['powerPS']<10)
sum(car_data1['powerPS']>500)
 