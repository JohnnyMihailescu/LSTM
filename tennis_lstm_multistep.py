import pandas
from time import strftime, gmtime
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
import glob
import math
from matplotlib import pyplot
import numpy
from keras.utils import plot_model
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')


def load_data(path):
    allFiles = glob.glob(path + "/*.csv")
    frame = pandas.DataFrame()
    frameList = []
    for dataFile in allFiles:
        df = pandas.read_csv(dataFile, header=0, parse_dates=True, index_col=0, squeeze=True,
                             usecols=['tourney_date', 'winner_name', 'loser_name', 'winner_rank', 'loser_rank'])
        frameList.append(df)
    series = pandas.concat(frameList)
    series = series.sort_index()
    return series

#convert raw match data to a time series of rank number for a player
def match_data_to_time_series(series, player_name):
    names = series[['winner_name', 'loser_name']]
    names_array = names.as_matrix().flatten()
    names_array_unique = numpy.unique(names_array)
    # print(names_array_unique)

    # iterate over dataframe to get time series for a player
    # print(series)
    time_series = list()
    for index, row in series.iterrows():
        if (row["winner_name"] == player_name and not math.isnan(row["winner_rank"])):
            time_series.append(row["winner_rank"])
        elif (row["loser_name"] == player_name and not math.isnan(row["loser_rank"])):
            time_series.append(row["loser_rank"])
    time_series = Series(time_series)
    return time_series

#frame a sequence as a supervised learning problem, with choice for how many time steps should be used per row
def timeseries_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols,names = list(), list()
    #input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    #forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i==0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    #aggregate
    agg = concat(cols, axis=1)
    agg.columns = names

    #drop NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def inverse_difference(last_ob, forecast):
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted
def inverse_transform(series, forecasts, scaler, n_test, has_diff):
    inverted = list()
    for i in range(len(forecasts)):
        forecast = numpy.array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        if has_diff:
            index = len(series) - n_test + i - 1
            last_ob = series.values[index]
            inv_diff = inverse_difference(last_ob, inv_scale)
            inv_scale = inv_diff
        else:
            inv_scale = inv_scale.tolist()
        inverted.append(inv_scale)
    return inverted

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def prepare_data(series, n_test, n_lag, n_seq, has_diff=False):
    raw_values = series.values
    raw_values = raw_values.reshape(len(raw_values), 1)
    if has_diff:
        diff_series = difference(raw_values, 1)
        diff_values = diff_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)
        raw_values = diff_values
    #rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(raw_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    #transform into supervised learning problem
    supervised = timeseries_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    #split into training and testing dataset
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test

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
def fit_lstm(trains, tests, raw_data, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    #reshape training into [samples, timesteps, features]
    X_trains = []
    y_trains = []
    for i in range(len(trains)):
        train = trains[i]
        X_train, y_train = train[:, 0:n_lag], train[:, n_lag:]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_trains.append(X_train)
        y_trains.append(y_train)

    #create netowork
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X_trains[0].shape[1], X_trains[0].shape[2]), stateful=True))
    model.add(Dense(y_trains[0].shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #fit (train on data)
    train_rmse, test_rmse = [], []
    for epoch in range(nb_epoch):
        mean_loss = []
        mean_acc = []
        for i in range(len(X_trains)):
            X = X_trains[i]
            y = y_trains[i]
            model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
            model.reset_states()
        #print("training accuracy = {}".format(numpy.mean(mean_acc)))
        #print("training loss = {}".format(numpy.mean(mean_loss)))
        #print("--------------------------------------")
    return model

#fitting and forecasting specifically for evaluating model strength
def test_lstm(trains, tests, raw_data, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    #reshape training into [samples, timesteps, features]
    X_trains = []
    y_trains = []
    for i in range(len(trains)):
        train = trains[i]
        X_train, y_train = train[:, 0:n_lag], train[:, n_lag:]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_trains.append(X_train)
        y_trains.append(y_train)

    #create netowork
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X_trains[0].shape[1], X_trains[0].shape[2]), stateful=True))
    model.add(Dense(y_trains[0].shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #fit (train on data)
    train_rmse, test_rmse = [], []
    for epoch in range(nb_epoch):
        mean_loss = []
        mean_acc = []
        for i in range(len(X_trains)):
            X = X_trains[i]
            y = y_trains[i]
            model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
            model.reset_states()
            raw_train = raw_data[-(len(train) + len(test) + 1): -len(test)]
            train_rmse.append(evaluate_model(model, raw_train, train, scaler, 0, n_batch))

            raw_test = raw_data[-(len(test + 1)):]
            test_rmse.append(evaluate_model(model, raw_test, test, scaler, 0, n_batch))
            model.reset_states()
        history = DataFrame()
        history['train'], history['test'] = train_rmse, test_rmse
        #print("training accuracy = {}".format(numpy.mean(mean_acc)))
        #print("training loss = {}".format(numpy.mean(mean_loss)))
        #print("--------------------------------------")
    return model



#make single forecast
def forecast_lstm(model, X, n_batch):
    X = X.reshape(1, 1, len(X))
    forecast = model.predict(X, batch_size=n_batch)
    return [x for x in forecast[0, :]]

#TODO should the training set be forecasted too? something to do with setting the state in the LSTM
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        forecast = forecast_lstm(model, X, n_batch)
        forecasts.append(forecast)
    return forecasts

def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = math.sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))

def plot_forecasts(series, forecasts, n_test):
    pyplot.plot(series.values)
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i])+1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='red')
    pyplot.show()

#used to get the rmse of the forecasts on a dataset
def evaluate_model(model, raw_data, scaled_data, scaler, offset, batch_size):

    X, y = scaled_data[:, 0:-1], scaled_data[:, -1]
    reshaped_X = X.reshape(len(X), 1, 1)
    prediction = model.predict(reshaped_X, batch_size=batch_size)
    predictions = []
    for i in range(len(prediction)):
        yhat = prediction[i, 0]
        yhat = invert_scale(scaler, X[i], yhat)
        if(has_diff):
            yhat = yhat + raw_data[i]
        predictions.append(yhat)
    rmse = math.sqrt(mean_squared_error(raw_data[1:], predictions))
    return rmse
#configuration parameters
has_diff = False
n_lag = 1
n_seq = 20
n_test = 40
n_epochs = 1500
n_batch = 1
n_neurons = 1
num_players = 10
time_series = []
scalers = []
trains = []
tests = []
player_names = ["Roger Federer", "Lleyton Hewitt", "Feliciano Lopez", "Richard Gasquet", "Rafael Nadal", "David Ferrer",
               "Mikhail Youzhny", "Novak Djokovic", "Radek Stepanek", "Tomas Berdych"]
#player_names = ["Tomas Berdych"]
path = r'C:\Users\John\Google Drive\Machine Learning\Python\LSTM\data'
#load dataset, loading all data from 2000 to 2016
series = load_data(path)
print("Series imported and sorted: " + strftime("%Y-%m-%d %H:%M:%S"))
#print(series)
for i in range(num_players):
    time_series.append(match_data_to_time_series(series, player_names[i]))
    #pyplot.plot(time_series[i])
    #pyplot.show()
print("Time series for player created " + strftime("%Y-%m-%d %H:%M:%S"))
#print(time_series)



#time_series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#convert to supervised learning
#rescale the data so it can be used in the lstm
for i in range(num_players):
    scaler, train, test = prepare_data(time_series[i], n_test, n_lag, n_seq, has_diff)
    scalers.append(scaler)
    trains.append(train)
    tests.append(test)
print("Time series transformed, partitioned and rescaled " + strftime("%Y-%m-%d %H:%M:%S"))
#fit the model TODO tune these parameters
lstm_model = fit_lstm(trains, n_lag, n_seq, n_batch, n_epochs, n_neurons)
print("Training on lstm done " + strftime("%Y-%m-%d %H:%M:%S"))
#forecast the entire training data to build up state for forecasting

forecasts = []
for i in range(num_players):
    forecast = make_forecasts(lstm_model, n_batch, trains[i], tests[i], n_lag, n_seq)
    forecast = inverse_transform(time_series[i], forecast, scalers[i], n_test+n_seq-1, has_diff)
    actual = [row[n_lag:] for row in tests[i]]
    actual = inverse_transform(time_series[i], actual, scalers[i], n_test+n_seq-1, has_diff)
    forecasts.append(forecast)
    evaluate_forecasts(actual, forecasts[i], n_lag, n_seq)
    plot_forecasts(time_series[i], forecasts[i], n_test+n_seq-1)
    print("Forecasting done " + strftime("%Y-%m-%d %H:%M:%S"))
    plot_model(lstm_model, to_file='tennis_model.png')



