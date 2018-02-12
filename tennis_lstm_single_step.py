from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import math
from matplotlib import pyplot
import numpy

#frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1,lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0,inplace=True)
    return df
#transform the scale of training and testing sets
def scale(train, test):
    #fit scaler
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train)
    #transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    #transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

#invert scaling operation
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

#reshape data into 3D matrix
#create LSTM model
#manually fit network to training data
def fit_lstm(train,batch_size,nb_epoch,neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

#make a one step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]


#load dataset
series = read_csv('atp_matches_2000.csv', header =0, parse_dates=True, index_col=0, squeeze=True, usecols=['tourney_date', 'winner_name', 'loser_name', 'winner_rank', 'loser_rank'])
series = series.sort_index()
#print(series)
names = series[['winner_name', 'loser_name']]
names_array = names.as_matrix().flatten()
names_array_unique = numpy.unique(names_array)
#print(names_array_unique)

#iterate over dataframe to get time series for a player
#print(series)
time_series = list()
for index, row in series.iterrows():
    if(row["winner_name"] == "Roger Federer" and not math.isnan(row["winner_rank"])):
        time_series.append(row["winner_rank"])
    elif(row["loser_name"] == "Roger Federer" and not math.isnan(row["loser_rank"])):
        time_series.append(row["loser_rank"])
time_series = Series(time_series)
#print(time_series)
pyplot.plot(time_series)
pyplot.show()
#convert to supervised learning
supervised = timeseries_to_supervised(time_series, 1)
#print(supervised)


#split data into training and testing sets
train,test = supervised.values[0:-21], supervised.values[-21:]
#print(train)
#print(test)

#rescale the data so it can be used in the lstm
scaler, train_scaled, test_scaled = scale(train, test)

#fit the model
lstm_model = fit_lstm(train_scaled, 1, 3000, 4)

#forecast the entire training data to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

#walk-forward validation on test data
predictions = list()
for i in range(len(test_scaled)):
    #make one-step forecast
    X,y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    #invert scaling
    yhat = invert_scale(scaler, X, yhat)
    #store forecast
    predictions.append(yhat)
    expected = time_series.values[len(train) + i]
    print('time = %d, predicted rank = %f, expected rank = %f ' % (i+1, yhat, expected))

#report performance
rmse = math.sqrt(mean_squared_error(time_series.values[-21:], predictions))
print('Test RMSE %.3f' % rmse)

pyplot.plot(time_series.values[-21:])
pyplot.plot(predictions)
pyplot.show()
