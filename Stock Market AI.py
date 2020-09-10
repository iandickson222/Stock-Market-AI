#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup


#get data
shiller_pe = requests.get("https://www.multpl.com/shiller-pe/table/by-month")
shiller_soup = BeautifulSoup(shiller_pe.content, 'html.parser')
shiller_table = shiller_soup.find(id="datatable")
shiller_dates, shiller_values = shiller_table.select(".left"), shiller_table.select(".right")
shiller_dates, shiller_values = [date.get_text() for date in shiller_dates], [value.get_text() for value in shiller_values]
shiller_dates.pop(0), shiller_values.pop(0)
shiller_values = [float(value.replace("\n", "").replace(',', '')) for value in shiller_values]
shiller_dates, shiller_values = shiller_dates[1500::-1], shiller_values[1500::-1]

prices = requests.get("https://www.multpl.com/s-p-500-historical-prices/table/by-month")
prices_soup = BeautifulSoup(prices.content, 'html.parser')
prices_table = prices_soup.find(id="datatable")
prices_dates, prices_values = prices_table.select(".left"), prices_table.select(".right")
prices_dates, prices_values = [date.get_text() for date in prices_dates], [value.get_text() for value in prices_values]
prices_dates.pop(0), prices_values.pop(0)
prices_values = [float(value.replace("\n", "").replace(',', '')) for value in prices_values]
prices_dates, prices_values = prices_dates[1500::-1], prices_values[1500::-1]


#prepare data
month_catagory = {"Jan": 0., "Feb": 1., "Mar": 2., "Apr": 3., "May": 4., "Jun": 5., "Jul": 6., 
                  "Aug": 7., "Sep": 8., "Oct": 9., "Nov": 10., "Dec": 11.}

prices_dates = [month_catagory[date.split()[0]] for date in prices_dates]

data = {"month": prices_dates, "prices": prices_values, "shiller PE": shiller_values}
df = pd.DataFrame(data)

months_ahead = 1
df['future'] = df['prices'].shift(-months_ahead)
df[['prices','future']] = df[['prices','future']].pct_change()
df.dropna(inplace=True)
df_mean = df.mean()
df_std = df.std()
df = (df - df_mean)/df_std
df.head(3)


#convert to numpy dataset
features = []
labels = []
for i in range(len(df)):
    ds = df.iloc[i].to_list()
    features.append([ds[0], ds[1], ds[2]])
    labels.append([ds[3]])
features = np.array(features)
labels = np.array(labels)
features_test = features.reshape(1500,3,1)
labels_test = labels.reshape(1500,1,1)
features_test = features_test[:-36]
labels_test = labels_test[:-36]


#build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(128, return_sequences=True, activation='relu'))
model.add(tf.keras.layers.LSTM(64, return_sequences=False, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])


#train model
model.fit(features_test, labels_test, epochs=5, validation_split=0.05)


#graph prediction
features_prediction = features.reshape(1500,3,1)

time=0
if time != 0:
    y = prices_values[-(time+12):-time]
    features_prediction = features_prediction[-(time+1):-time]
    
else:
    y = prices_values[-12:]
    features_prediction = features_prediction[-1:]

prediction_percentage = model.predict(features_prediction)
percentage_change = prediction_percentage*df_std['prices'] + df_mean['prices']
prediction = y[-1]*percentage_change + y[-1]

x = range(12)
plt.scatter(x, y, marker='o')

if time != 0:
    plt.scatter([12], [prices_values[-time]], color='blue')
else:
    plt.scatter(None, None, color='blue')
    
plt.scatter([12], [prediction], color='green')
plt.legend(['Historical Prices','Actual Price', 'Predicted Price'])
plt.title("SP 500")
plt.xlabel("Month")
plt.ylabel("Price")

print(f'Predicted percent change: {percentage_change[0][0]*100}')




