# Source for most of it: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from os import path
import sys

def dataset_filter(df, companies_file):
    # filter dataset
    companies = pd.read_csv(companies_file)
    companies_lst = companies.T.values.tolist()[0]
    # filter the companies
    company_filter = df['Name'].isin(companies_lst)
    df = df[company_filter]

    # dates per company
    dates_num = len(df['date'].unique())
    # filter out companies with not the same time series
    freqs = df['Name'].value_counts()
    freqs_index = freqs[freqs == dates_num].index
    freqs_filter = df['Name'].isin(freqs_index)
    df = df[freqs_filter].reset_index(drop=True)
    df = df.drop(['date', 'Name'], axis=1)
    return df

def df_to_supervised(df, n_in=1, n_out=1, vars_output=['close'], dropnan=True):
    # convert time series into supervised learning problem
    if not vars_output:
        print('Define variables of the dataset for the output (prediction)')
        sys.exit(1)
    n_out = len(vars_output)
    vars_output_index = []
    for var_output in vars_output:
        vars_output_index.append(
            df.columns.get_loc(var_output))

    n_vars = df.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df[vars_output].shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in vars_output_index]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in vars_output_index]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    agg.reset_index(drop=True, inplace=True)
    return agg

def difference(dataset, columns, interval=1):
    # create a differenced series
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.DataFrame(diff, columns=columns)

def inverse_difference(history, yhat, interval=1):
    # invert differenced value
    return yhat + history[-interval]

def scale(train, test):
    # scale train and test data to [-1, 1]

    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    #train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    #test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

def invert_scale(scaler, X, value):
    # inverse scaling for a forecasted value
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
