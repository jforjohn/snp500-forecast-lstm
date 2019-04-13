import pandas as pd
import argparse
from MyPreprocessing import *
from time import time
import json
from os import path
import sys
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)

##
def load_config_file(nfile, abspath=False):
    """
    Read the configuration from a json file

    :param abspath:
    :param nfile:
    :return:
    """
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)
##
if __name__ == '__main__':
    # Loads config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="config",
        help="specify the location of the config file"
    )
    args, _ = parser.parse_known_args()

    config_file = args.config
    config = load_config_file(config_file, './')

    config_data = config['data']
    
    ##
    data_dir = config_data.get('data_dir')
    dataset = config_data.get('dataset')
    path_data = path.join(data_dir, dataset)

    try:
        df = pd.read_csv(path_data, header=0)
    except FileNotFoundError:
        print("Dataset '%s' cannot be found in the path %s" %(dataset, path_data))
        sys.exit(1)

# filter in the InfoTech sector
companies_file = path.join(config_data.get('data_dir'), 'InfoTechCompanies.csv')
df = dataset_filter(df, companies_file)
# configure
config_preprocessing = config['preprocessing']

lag = config_preprocessing.get('lag', 7)
no_multisteps = config_preprocessing.get('no_multisteps', 1)
test_size = config_preprocessing.get('test_size', 0.1)

if config_preprocessing.get('multivariate'):
    data = df
    vars_output = config_preprocessing.get('vars_output', 'close')
else:
    data = pd.DataFrame(df['close'], columns=['close'])
    vars_output = ['close']
# prepare data
#prep_data = prepare_data(data, n_test, n_lag, n_seq)
#print('Train: %s, Test: %s' % (train.shape, test.shape))
# transform data to be stationary

if config_preprocessing.get('stationary'):
    data = difference(data.values, data.columns, 1)

prep_data = df_to_supervised(data, lag, no_multisteps, vars_output)

data_size = prep_data.shape[0]
trainset_offset = int((1-test_size)*data_size)
train_set, test_set = prep_data.loc[:trainset_offset,:], prep_data.loc[trainset_offset:,:]

scaler, train_scaled, test_scaled = scale(train_set, test_set)
