import tensorflow as tf
import tflearn
import numpy as np
from tensorflow.contrib import rnn
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import DataFrame
from pandas import concat
from pandas import Series


# def timeseries_to_superviser(data, lag=1):
#     df = DataFrame(data)
#     columns = [df.shift(i) for i in range(1,lag+1)]
#     columns.append(df)
#     df = concat(columns, axis = 1)
#     df.fillna(0, inplace = True)
#     return df
#

def parser(x):
    return datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
series = read_csv('bitcoin_ticker.csv', header = 0, index_col = 'created_at', parse_dates = [12], date_parser = parser)

# usecols = lambda x : x in ['rpt_key','created_at','updated_at']
fields = ['rpt_key', 'updated_at']

series = series[series.rpt_key == 'btc_krw']
drop_column = ['date_id', 'datetime_id', 'market','diff_24h','diff_per_24h','bid','ask','low','high', 'volume', 'updated_at','rpt_key']
series = series.drop(drop_column, axis = 1)
print(series.head())
#series.plot()
#pyplot.show()

from sklearn import preprocessing

X = series.values
X = X.reshape(len(X),1)
scaler = preprocessing.MinMaxScaler(feature_range = (-1,1))
scaler = scaler.fit(X)
scaled_X = scaler.transform(X)

inverted_X = scaler.inverse_transform(scaled_X)
inverted_series = Series(inverted_X[:, 0])
print(inverted_series.head())

train, test = X[0:round(len(X)*0.1)], X[-round(len(X)*0.1):]

X,y = train[:,0:-1], train[:, -1]
X = X.reshape(X.shape[0], 1, X.shape[1])

layer = LSTM(neurous, batch_input_shape = (batch_size, X.shape[1], X.shape[2], stateful = True)

model = Sequential()
model.add(layer)
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
