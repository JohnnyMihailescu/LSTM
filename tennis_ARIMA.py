from time import strftime, gmtime
import warnings
import numpy
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series
import pandas
import glob
import math
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
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
def match_data_to_time_series(series, player_name):
    names = series[['winner_name', 'loser_name']]
    names_array = names.as_matrix().flatten()
    names_array_unique = numpy.unique(names_array)
    # print(names_array_unique)

    # iterate over dataframe to get time series for a player
    # print(series)
    time_series = pandas.DataFrame(columns=["rank_points"])
    for index, row in series.iterrows():
        if row["winner_name"] == player_name and not math.isnan(row["winner_rank_points"]):
            entry = row
            entry["rank_points"] = row["winner_rank_points"]
            entry = entry.drop(["winner_rank_points", "loser_rank_points", "winner_name", "loser_name"])
            entry.name = index
            time_series = time_series.append(entry)
        elif row["loser_name"] == player_name and not math.isnan(row["loser_rank_points"]):
            entry = row
            entry["rank_points"] = row["loser_rank_points"]
            entry = entry.drop(["winner_rank_points", "loser_rank_points", "winner_name", "loser_name"])
            entry.name = index
            time_series = time_series.append(entry)
    return time_series
def evaluate_forecasts(test, forecasts, n_seq, player_name):
    rmse_list = []
    actual = []
    file = open("{}_Arima.txt".format(player_name), "w+")
    for i in range(n_seq):
        actual = test[i:i+len(forecasts)]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = math.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        file.write('t+%d RMSE: %f MAE: %f\n' % ((i+1), rmse, mae))
        #print('t+%d RMSE: %f MAE: %f' % ((i+1), rmse, mae))
        rmse_list.append(rmse)
    return rmse_list

def plot_forecasts(series, forecasts, n_test, player_name):
    pyplot.plot(series, label="Actual Ranking Points")
    pyplot.title("Ranking points of {} over time with ARIMA".format(player_name))
    pyplot.xlabel("Timesteps")
    pyplot.ylabel("Ranking points")
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i])+1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = numpy.append(series[off_s], forecasts[i])
        if i==0:
            pyplot.plot(xaxis, yaxis, color='red', label="Forecasted Ranking Points")
        else:
            pyplot.plot(xaxis, yaxis, color='red')
    pyplot.legend()
    pyplot.show()
    #pyplot.savefig('{}_{}_{}'.format(player_name, n_epochs, n_neurons))

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)
def evaluate_arima(series, order,n_test,n_seq, n_lag, player_name):
    train, test = series[0:-(n_test++n_seq-1)], series[-(n_test++n_seq-1):]
    history = [x for x in train]
    forecasts = []
    for i in range(len(test)-n_seq):
        model = ARIMA(history, order = order)
        model_fit = model.fit(disp=0)
        output = model_fit.predict(start=i + len(train), end=i + len(train) + n_seq-1, typ='levels')
        forecasts.append(output)
        history.append(test[i])
    errors = evaluate_forecasts(test,forecasts,n_seq, player_name)
    return errors
def evaluate_model(series, p_values, d_values, q_values, n_test,n_seq, n_lag, player_name):
    series = series.astype('float32')
    min_error, best_order = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    errors = evaluate_arima(series, order, n_test,n_seq, n_lag, player_name)
                    if errors[19] < min_error:
                        min_error, best_order = errors[19], order
                    print("ARMIA{} error:{} ".format(order, errors[19]))
                except:
                    continue

    print("Best ARIMA{} error={}".format(best_order, min_error))

def run_test():
    time_series = []
    num_players = 1
    n_test = 40
    n_seq = 20
    n_lag = 10
    player_names = ["Roger Federer", "Lleyton Hewitt", "Feliciano Lopez", "Richard Gasquet", "Rafael Nadal",
                    "David Ferrer",
                    "Mikhail Youzhny", "Novak Djokovic", "Radek Stepanek", "Tomas Berdych"]
    player_names = ["Marco Chiudinelli"]
    path = r'C:\Users\John\Google Drive\Machine Learning\Python\LSTM\data'
    # load dataset, loading all data from 2000 to 2016
    series = load_data(path)
    print("Series imported and sorted: " + strftime("%Y-%m-%d %H:%M:%S"))
    # print(series)
    for i in range(num_players):
        time_series.append(match_data_to_time_series(series, player_names[i]))
    print("Time series for player created " + strftime("%Y-%m-%d %H:%M:%S"))

    p_values = [0,1,2,4,6,8,10]
    d_values = range(1, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")
    evaluate_model(time_series[0].values, p_values, d_values, q_values, n_test, n_seq, n_lag, player_names[0])

def run_forecast():
    time_series = []
    num_players = 1
    n_test = 40
    n_seq = 20
    n_lag = 10
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
    #pyplot.plot(time_series[0])
    #pyplot.show()
    print("Time series for player created " + strftime("%Y-%m-%d %H:%M:%S"))
    #autocorrelation_plot(time_series[0])
    #pyplot.show()

    X = time_series[0].values
    train, test = X[0:-(n_test++n_seq-1)], X[-(n_test+n_seq-1):]
    history = [x for x in train]
    predictions = []
    forecasts = []
    for i in range(len(test)-n_seq):
        model = ARIMA(history, order=(6, 2, 2))
        model_fit = model.fit(disp=0)
        output = model_fit.predict(start=i+len(train), end=i+len(train)+n_seq-1, typ='levels')
        yhat = output[0]
        predictions.append(yhat)
        forecasts.append(output)
        observed = test[i]
        history.append(observed)
        #print('predicted={}, expected={}'.format(yhat,observed))
    errors = evaluate_forecasts(test, forecasts, n_seq, player_names[0])
    print('Test mse: {}'.format(errors[19]))
    #pyplot.plot(test)
    #pyplot.plot(predictions, color='red')
    #pyplot.show()

    plot_forecasts(X, forecasts, len(test), player_names[0])
    #residuals = DataFrame(model_fit.resid)
    #residuals.plot()
    #pyplot.show()
    #residuals.plot(kind='kde')
    #pyplot.show()
    #print(residuals.describe())


run_test()