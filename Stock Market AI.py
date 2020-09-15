#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup


def get_data(url, length):
    url_html = requests.get(url)
    soup = BeautifulSoup(url_html.content, 'html.parser')
    table = soup.find(id="datatable")
    dates, values = table.select(".left"), table.select(".right")
    dates, values = [date.get_text() for date in dates], [value.get_text() for value in values]
    dates.pop(0), values.pop(0)
    dates.pop(0), values.pop(0)
    values = [float(value.replace("\n", "").replace(',', '')) for value in values]
    dates, values = dates[length::-1], values[length::-1]
    return dates, values

prices_dates, prices_values = get_data("https://www.multpl.com/s-p-500-historical-prices/table/by-month", 1500)
shiller_dates, shiller_values = get_data("https://www.multpl.com/shiller-pe/table/by-month", 1500)


def prepare_data(dates, prices, shiller):   
    month = {"Jan": 0., "Feb": 1., "Mar": 2., "Apr": 3., "May": 4., "Jun": 5., "Jul": 6., "Aug": 7., "Sep": 8., "Oct": 9., "Nov": 10., "Dec": 11.}
    dates = [month[date.split()[0]] for date in dates]

    data = {"Month": dates, "Prices": prices, "Shiller PE": shiller}
    df = pd.DataFrame(data)
    df['Prices'] = df['Prices'].pct_change()
    df.dropna(inplace=True)
    return df

df = prepare_data(prices_dates, prices_values, shiller_values)


def standardize_data(data):
    data = (data - data.mean())/data.std()
    return data

standardized_df = standardize_data(df)    


def split_data(data):
    features = data[:,:-1,:]   
    labels = data[:,-1:, :]  
    label = tf.stack([labels[:, :, 1]], axis=-1)
    return features, label


def preprocess_data(data, sequence_length = 10):
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data = data,
        targets = None,
        sequence_length = sequence_length,
        batch_size= 12,
        shuffle = False,
    )
    return ds.map(split_data)

ds = preprocess_data(standardized_df)


def seperate_dataset(data):
    train = data.take(len(data) - 2)
    validation = data.skip(len(data) - 2)   
    return train, validation

train_ds, validation_ds = seperate_dataset(ds)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences = True, activation = 'relu')))
model.add(tf.keras.layers.Dense(1))
model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam(), metrics = ['mae'])


early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 2, mode = 'min')
model.fit(train_ds, epochs = 50, validation_data = validation_ds, callbacks = [early_stopping])


months = 12
predictions = []
for i in range(0, months):
    if i == 0:
        test_ds = standardized_df[-9:].values.reshape(-1, 9, 3)
    else:       
        test_ds = standardized_df[-i-9:-i].values.reshape(-1, 9, 3)
           
    prediction = model.predict(test_ds)
    prediction = prediction*df['Prices'].std() + df['Prices'].mean()
    prediction = prediction*prices_values[-i-1] + prices_values[-i-1]
    predictions.append(prediction[0,-1,0])

predictions = predictions[::-1]


plt.scatter(range(0, months - 1), prices_values[-(months - 1):], label='Actual Prices')
plt.scatter(range(0, months), predictions, label='Prediction')
plt.title('Stock Market Predictions')
plt.xlabel('Months')
plt.ylabel('Prices')
plt.legend()
plt.show()
print(f"Prediction: {predictions[-1]}")





