import tensorflow as tf
import tflearn
import numpy as np
from tensorflow.contrib import rnn
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt

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

X = series.values
train, test = X[0:round(len(X)*0.1)], X[-round(len(X)*0.1):]


history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()
