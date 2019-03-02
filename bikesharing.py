# -*- coding: utf-8 -*-
"""
Created on: Wed Feb 27 19:31:56 2019

@author: Marcin Tutajewski
"""
"""
datetime - hourly date + timestamp  
season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
holiday - whether the day is considered a holiday
workingday - whether the day is neither a weekend nor holiday
weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
temp - temperature in Celsius
atemp - "feels like" temperature in Celsius
humidity - relative humidity
windspeed - wind speed
casual - number of non-registered user rentals initiated
registered - number of registered user rentals initiated
count - number of total rentals
"""


#1. Importing modules and loading the data
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_style('dark', {'axes.grid':False})
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('train.csv', parse_dates=[0]) 
test = pd.read_csv('test.csv', parse_dates=[0])
data = data.append(test,sort=False)

#2. Normalizing data, converting columns
data['count'].hist(bins=20)
plt.show()
data['count'] = np.log(data['count']+1)
data['count'].hist(bins=20)
plt.show()

data['year'], data['month'], data['day'], data['weekday'], data['hour'] = data['datetime'].dt.year, data['datetime'].dt.month, data['datetime'].dt.day, data['datetime'].dt.dayofweek, data['datetime'].dt.hour
data.sort_values('datetime', inplace=True)


#3. Plotting data to look for potential outliers in the data (peak days) -- EDA
fig, ax = plt.subplots(nrows=3, ncols=2)
fig.set_size_inches(15,12)

sns.boxplot(data=data, x='year', y='count', palette='muted', ax=ax[0][0])
ax[0][0].set(xlabel='Year', ylabel='Bikes Rented')
#More bikes rented in 2012

sns.boxplot(data=data, x='season', y='count', palette='muted', ax=ax[0][1])
ax[0][1].set(xlabel='Season', ylabel='Bikes Rented')
#Most of the bikes are rented over the summer and autumn.
#People are not interested in them that much in Spring.

sns.boxplot(data=data, x='weekday', y='count', palette='muted', ax=ax[1][0])
ax[1][0].set(xlabel='Day of the Week', ylabel='Bikes Rented')
#The number of rentals seems to be quite equally distributed #over the days of the week. There are, however, some days with a #large number of outliers like Tuesday and Wednesday

sns.boxplot(data=data, x='hour', y='count', palette='muted', ax=ax[1][1])
ax[1][1].set(xlabel='Hour', ylabel='Bikes Rented')
#People mostly use bikes for getting to and from work??

sns.boxplot(data=data, x='windspeed', y='count', palette='muted', ax=ax[2][0])
ax[2][0].set(xlabel='Speed of the Wind', ylabel='Bikes Rented')
#Does not seem like the windspeed affects decision about renting #a bike. It needs to be stressed out that weather conditions are #not equally distributed!

sns.boxplot(data=data, x='humidity', y='count', palette='muted', ax=ax[2][1])
ax[2][1].set(xlabel='Humidity in %', ylabel='Bikes Rented')
#People tend to use bikes more often in lower humidity, but this #may also be impacted by the climate rather than their own #decisions.

plt.show()
#####################################################################
#Diving into two sets and dropping some unnecessary features
test = data[data['count'].isnull()]
data = data[~data['count'].isnull()]

to_drop = ['casual','registered','datetime','count']
train_features = [c for c in data.columns if c not in to_drop]


#4. Machine Learning part
train, validation = train_test_split(data, random_state=7, test_size=0.25)
print(train.shape, validation.shape)

#I have already tested a few classifiers and it turns out that these #three work best for the problem..

#Decision Tree
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=10, random_state=7)
tree.fit(train[train_features], train['count'])
treePrediction = tree.predict(validation[train_features])
treeScore = mean_squared_error(validation['count'],
                               treePrediction)**(1/2) #math.sqrt

#Random Forest
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=200, max_depth=10,
                            random_state=7)
rfr.fit(train[train_features], train['count'])
rfrPrediction = rfr.predict(validation[train_features])
rfrScore = mean_squared_error(validation['count'],
                              rfrPrediction)**(1/2)

#Stochastic Gradient boosting (subsample<1)
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=200,learning_rate=0.1, 
                                subsample=0.6,random_state=7)

gbr.fit(train[train_features], train['count'])
gbrPrediction = gbr.predict(validation[train_features])
gbrScore = mean_squared_error(validation['count'],
                              gbrPrediction)**(1/2)

#Dataframe for the results
models = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'Gradient Boosting'],
    'Score': [treeScore, rfrScore, gbrScore]})
print(models.sort_values(by=['Score'], ascending=True))

#Gradient Boost is the winner

#Submission file
test['count'] = np.exp(gbr.predict(test[train_features]))
test[['datetime', 'count']].to_csv('submission.csv', index=False)












