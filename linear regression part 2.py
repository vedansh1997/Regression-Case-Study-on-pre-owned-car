# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 23:38:02 2023

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
    
## visiualizing parameter
'''Age'''
sns.distplot(car_data1['Age'],kde=False)
sns.boxenplot(car_data1['Age'])
# Price
sns.distplot(car_data1['price'])
sns.boxplot(car_data1['price'])
'''powerPS'''
sns.distplot(car_data1['powerPS'])
sns.boxplot(car_data1['powerPS'])
''' now if you see all the histogram and box plot it is better then the previous
  one'''
### visualization age vs price
sns.regplot(data=car_data1,x='Age',y='price',scatter=True,fit_reg=False)
# car price higher of newer cars
# as the age of car increases price decreases, some are vintage exception
# some car price increase with increase of age
''' price Vs powerPS '''
sns.regplot(data=car_data1,x='powerPS',y='price',fit_reg=False,scatter=True)
### power increases as the price increase
''' now we are going to check what are the effect of the other  variable  on price'''
car_data1['seller'].value_counts()
pd.crosstab(car_data1['seller'], columns='counts',normalize=True)
sns.countplot(x='seller',data=car_data1)
### fewer cars are commercial=> insignificant
## now test for abtest
car_data1['abtest'].value_counts()
pd.crosstab(car_data['abtest'], columns='counts',normalize=True)
sns.countplot(data=car_data1,x='abtest')
### approximately equally distributed
sns.boxenplot(x='abtest',y='price',data=car_data1)
# as you see the box plot both are same
# so they are insignificant

### variable Vehicle type
car_data1['vehicleType'].value_counts()
pd.crosstab(car_data['vehicleType'], columns='counts',normalize=True)
sns.countplot(data=car_data1,x='vehicleType')
sns.boxplot(data=car_data1,x='vehicleType',y='price')
# 8 types of vehicle in the data such as limousine,Suv,Bus,Small car,Station Wagon etc.
# so here vehicle type affects price


# now for gearbox
car_data1.columns
car_data1['gearbox'].value_counts()
pd.crosstab(car_data1['gearbox'], columns='count',normalize=True)
sns.countplot(data=car_data1,x='gearbox')
sns.boxplot(data=car_data1,x='gearbox',y='price')
# gear box affects prices


# variablemodel
car_data1['model'].value_counts()
pd.crosstab(car_data1['model'], columns='counts',normalize=True)
sns.countplot(data=car_data1,x='model')


# now for kilometer
car_data1['kilometer'].value_counts()
pd.crosstab(car_data1['kilometer'], columns='count',normalize=True)
sns.countplot(data=car_data1,x='kilometer')
sns.boxplot(data=car_data1,x='kilometer',y='price')
car_data1['kilometer'].describe()
sns.regplot(data=car_data1,x='kilometer',y='price',fit_reg=False,scatter=True)
# considered in modelling


# variable fuel type
car_data1['fuelType'].value_counts()
pd.crosstab(car_data1['fuelType'], columns='count',normalize=True)
sns.countplot(data=car_data1,x='fuelType')
sns.boxplot(data=car_data1,x='fuelType',y='price')
# fuel type affects price

# variable brand
car_data1['brand'].value_counts()
pd.crosstab(car_data1['brand'], columns='count',normalize=True)
sns.countplot(data=car_data1,x='brand')
sns.boxplot(data=car_data1,x='brand',y='price')
# cars are distributed over various brand 
# considered them for modelling


# variable notRepairedDamage
# yes- car is damaged but not rectified
# no - car was damaged but has been ractified
car_data1['notRepairedDamage'].value_counts()
pd.crosstab(car_data1['notRepairedDamage'], columns='count',normalize=True)
sns.countplot(data=car_data1,x='notRepairedDamage')
sns.boxplot(data=car_data1,x='notRepairedDamage',y='price')
# price is affected by damage

# remove insignificant variable
col1=['abtest','seller','offerType']
car_data1=car_data1.drop(col1,axis=1)
car_data2=car_data1.copy()

### correlation
car_select=car_data1.select_dtypes(exclude=[object])
correlation=car_select.corr()
correlation=round(correlation,3)
car_select.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]
