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


def timeseries_to_superviser(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1,lag+1)]
    columns.append(df)
    df = concat(columns, axis = 1)
    df.fillna(0, inplace = True)
    return df


def parser(x):
    return datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
series = read_csv('bitcoin_ticker.csv', header = 0, index_col = 'created_at', parse_dates = [12], date_parser = parser)

# usecols = lambda x : x in ['rpt_key','created_at','updated_at']
fields = ['rpt_key', 'updated_at']

series = series[series.rpt_key == 'btc_krw']
drop_column = ['date_id', 'datetime_id', 'market','diff_24h','diff_per_24h','bid','ask','low','high', 'volume', 'updated_at','rpt_key']
series = series.drop(drop_column, axis = 1)
series_super = timeseries_to_superviser(series)
print(series.head())
#series.plot()
#pyplot.show()

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

differenced = difference(series.values, 1)
print(differenced.head())

inverted = list()
for i in range(len(differenced)):
    value = inverse_difference(series, differenced[i], len(series)-i)
    inverted.append(value)
inverted = Series(inverted)
print(inverted.head())
