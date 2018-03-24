import pandas
from time import strftime, gmtime
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series
import time
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
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
                             usecols=['tourney_date', 'winner_name', 'loser_name', 'winner_rank_points', 'loser_rank_points'])
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
        if (row["winner_name"] == player_name and not math.isnan(row["winner_rank_points"])):
            time_series.append(row["winner_rank_points"])
        elif (row["loser_name"] == player_name and not math.isnan(row["loser_rank_points"])):
            time_series.append(row["loser_rank_points"])
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
    X = numpy.asarray(X)
    new_row = [x for x in X] + [numpy.asarray(value)]
    array = new_row.reshape(1, len(new_row))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

#reshape data into 3D matrix
#create LSTM model
#manually fit network to training data
def fit_lstm(trains, n_lag, n_seq, n_batch, nb_epoch, n_neurons, layers):
    #reshape training into [samples, timesteps, features]
    X_trains = []
    y_trains = []
    for i in range(len(trains)):
        train = trains[i]
        X_train, y_train = train[:, 0:n_lag], train[:, n_lag:]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_trains.append(X_train)
        y_trains.append(y_train)
    return_sequences = layers > 1
    #create netowork
    model = Sequential()
    model.add(LSTM(n_neurons, return_sequences=return_sequences, batch_input_shape=(n_batch, X_trains[0].shape[1], X_trains[0].shape[2]), stateful=True))
    for i in range(layers-1):
        return_sequences = layers-1-i > 1
        model.add(LSTM(n_neurons, return_sequences=return_sequences))
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

#make single forecast
def forecast_lstm(model, X, n_batch):
    X = X.reshape(1, len(X), 1)
    forecast = model.predict(X, batch_size=n_batch)
    return [x for x in forecast[0, :]]


def make_forecasts(model, n_batch, test, n_lag):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        forecast = forecast_lstm(model, X, n_batch)
        forecasts.append(forecast)
    return forecasts

def evaluate_forecasts(test, forecasts, n_lag, n_seq, player_name, epoch_number, neurons, layers, start_time):
    rmse_list = []
    time_finished = time.time()
    elapsed_time = time_finished-start_time
    #file = open("{}_{}epochs_{}neurons_{}lag_{}layers.txt".format(player_name, epoch_number, neurons, n_lag, layers), "w+")
    time_string = "Time elapsed: {}".format(str(timedelta(seconds=round(elapsed_time))))
    #file.write(time_string)
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = math.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        #file.write('t+%d RMSE: %f MAE: %f\n' % ((i+1), rmse, mae))
        #print('t+%d RMSE: %f' % ((i+1), rmse))
        rmse_list.append(rmse)

    return rmse_list

def plot_forecasts(series, forecasts, n_test, player_name, n_seq, n_epochs, n_neurons, layers):
    pyplot.plot(series.values, label='Actual Ranking Points')
    pyplot.title("Ranking points of {} over time{} {}".format(player_name, n_neurons, layers))
    pyplot.xlabel("Timesteps")
    pyplot.ylabel("Ranking points")
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i])+1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        if i == 0:
            pyplot.plot(xaxis, yaxis, color='red', label='Forecasted Ranking Points')
        else:
            pyplot.plot(xaxis, yaxis, color='red')
    axes = pyplot.gca()
    #axes.set_xlim([off_s, off_e+n_seq])
    pyplot.legend()
    pyplot.show()
    #pyplot.savefig('{}_{}_{}'.format(player_name, n_epochs, n_neurons))
#do a forecast
def forecast():
    #get start time
    start_time = time.time()
    # configuration parameters
    has_diff = False
    n_lag = 10
    n_seq = 20
    n_test = 40
    n_epochs = 100
    n_batch = 1
    n_neurons = 3
    layers = 1
    num_players = 1
    time_series = []
    scalers = []
    trains = []
    tests = []
    player_names = ["Roger Federer", "Lleyton Hewitt", "Feliciano Lopez", "Richard Gasquet", "Rafael Nadal", "David Ferrer",
                    "Mikhail Youzhny", "Novak Djokovic", "Radek Stepanek", "Tomas Berdych"]
    player_names = ["Marco Chiudinelli"]
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
    #fit the model
    lstm_model = fit_lstm(trains, n_lag, n_seq, n_batch, n_epochs, n_neurons, layers)
    print("Training on lstm done " + strftime("%Y-%m-%d %H:%M:%S"))
    #forecast the entire training data to build up state for forecasting

    forecasts = []
    for i in range(num_players):
        forecast = make_forecasts(lstm_model, n_batch, tests[i], n_lag)
        forecast = inverse_transform(time_series[i], forecast, scalers[i], n_test+n_seq-1, has_diff)
        actual = [row[n_lag:] for row in tests[i]]
        actual = inverse_transform(time_series[i], actual, scalers[i], n_test+n_seq-1, has_diff)
        forecasts.append(forecast)
        evaluate_forecasts(actual, forecasts[i], n_lag, n_seq, player_names[0], n_epochs, n_neurons, layers, start_time)
        plot_forecasts(time_series[i], forecasts[i], n_test+n_seq-1, player_names[i], n_seq, n_epochs, n_neurons, layers)
        print("Forecasting done " + strftime("%Y-%m-%d %H:%M:%S"))
        #plot_model(lstm_model, to_file='tennis_model.png')


#used to get the rmse of the forecasts on a dataset
def evaluate_model(model, raw_data, scaled_data, scaler, offset, batch_size, n_seq, n_lag, n_test, time_series, has_diff, player_name, n_epochs, neurons, layers, start_time):
    X, y = scaled_data[:, 0:n_lag], scaled_data[:, n_lag:]
    reshaped_X = X.reshape(X.shape[0], X.shape[1], 1)
    predictions = list()
    for i in range(reshaped_X.shape[0]):
        input = reshaped_X[i, :, :]
        forecast = forecast_lstm(model, input, batch_size)
        predictions.append(forecast)
    predictions = inverse_transform(time_series, predictions, scaler, n_test+n_seq-1, has_diff)
    rmse = evaluate_forecasts(raw_data, predictions, n_lag, n_seq, player_name, n_epochs, neurons, layers, start_time)
    return rmse

#fitting and forecasting specifically for evaluating model strength
def test_lstm(train, test, raw_data, scaler, n_lag, n_seq, n_test, n_batch, nb_epoch, n_neurons, time_series, has_diff, layers, player_name, start_time):
    #reshape training into [samples, timesteps, features]

    X_train, y_train = train[:, 0:n_lag], train[:, n_lag:]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    return_sequences = layers>1
    #create netowork
    model = Sequential()
    model.add(LSTM(n_neurons, return_sequences=return_sequences, batch_input_shape=(n_batch, X_train.shape[1], X_train.shape[2]), stateful=True))
    for i in range(layers-1):
        return_sequences = layers-i-1 > 1
        model.add(LSTM(n_neurons, return_sequences=return_sequences))
    model.add(Dense(y_train.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #fit (train on data)
    train_rmse, test_rmse = [], []
    histories = []
    for epoch in range(nb_epoch):
        mean_loss = []
        mean_acc = []
        model.fit(X_train, y_train, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
        actual_train = [row[n_lag:] for row in train]
        actual_train = inverse_transform(time_series, actual_train, scaler, n_test+n_seq-1, has_diff)
        train_rmse.append(evaluate_model(model, actual_train, train, scaler, 0, n_batch, n_seq, n_lag, n_test, time_series, has_diff, player_name, nb_epoch, n_neurons, layers, start_time))

        actual_test = [row[n_lag:] for row in test]
        actual_test = inverse_transform(time_series, actual_test, scaler, n_test+n_seq-1, has_diff)
        test_rmse.append(evaluate_model(model, actual_test, test, scaler, 0, n_batch, n_seq, n_lag, n_test, time_series, has_diff, player_name, nb_epoch, n_neurons, layers, start_time))
        model.reset_states()
        #print("training accuracy = {}".format(numpy.mean(mean_acc)))
        #print("training loss = {}".format(numpy.mean(mean_loss)))
        #print("--------------------------------------")
    for i in range(len(train_rmse[0])):
        history = DataFrame()
        history['train'], history['test'] = [row[0] for row in train_rmse], [row[0] for row in test_rmse]
        histories.append(history)
    return histories
#perform diagnostic test
def run_test():
    start_time = time.time()
    repeats = 5
    n_batch = 1
    n_epochs = 1000
    n_neurons = 3
    n_test = 40
    n_lag = 10
    n_seq = 20
    has_diff = False
    layers = 3
    player_name = "Roger Federer"
    path = r'C:\Users\John\Google Drive\Machine Learning\Python\LSTM\data'
    # loa d dataset, loading all data from 2000 to 2016
    series = load_data(path)
    print("Series imported and sorted: " + strftime("%Y-%m-%d %H:%M:%S"))
    # print(series)
    time_series = match_data_to_time_series(series, player_name)
        # pyplot.plot(time_series[i])
        # pyplot.show()
    print("Time series for player created " + strftime("%Y-%m-%d %H:%M:%S"))
    # print(time_series)
    scaler, train, test = prepare_data(time_series, n_test, n_lag, n_seq, has_diff)
    for i in range(repeats):
        histories = test_lstm(train, test, time_series.values, scaler, n_lag, n_seq, n_test, n_batch, n_epochs, n_neurons, time_series, has_diff, layers, player_name, start_time)
        pyplot.plot(histories[0]['train'], color = 'blue')
        pyplot.plot(histories[0]['test'], color = 'orange')
        print('%d) TrainRMSE=%f, TestRMSE=%f' % (i, histories[0]['train'].iloc[-1], histories[0]['test'].iloc[-1]))
    pyplot.show(block=False)
    pyplot.savefig('epochs_diagnostic_{}_{}3layers_neurons5repeats.png'.format(n_epochs, n_neurons))
    print("Finished " + strftime("%Y-%m-%d %H:%M:%S"))

forecast()