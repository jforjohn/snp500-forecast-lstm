import pandas as pd
import argparse
from MyPreprocessing import *
from time import time
import json
from os import path
import sys
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.optimizers import RMSprop, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import load_model
from MyModels import *


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)

##
def MainLauncher(config):
    print(config)

    # Loads config
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="config",
        help="specify the location of the config file"
    )
    args, _ = parser.parse_known_args()

    config_file = args.config
    #config = load_config_file(config_file, './')
    '''

    config_data = config['data']

    ##
    data_dir = config_data.get('data_dir')
    dataset = config_data.get('dataset')
    path_data = path.join(data_dir, dataset)
    output_dir = config_data.get('output_dir')
    
    try:
        df = pd.read_csv(path_data, header=0)
    except FileNotFoundError:
        print("Dataset '%s' cannot be found in the path %s" %(dataset, path_data))
        sys.exit(1)

    # filter in the InfoTech sector
    companies_file = path.join(config_data.get('data_dir'), 'InfoTechCompanies.csv')
    per_company_groups, df = dataset_filter(df, companies_file)
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

    # data size of each group is the number of their unique dates: 1259
    data_size_per_company = 1259
    # company offset
    offset = int((1-test_size)*data_size_per_company)
    no_outputs = len(vars_output) * no_multisteps

    print('offset', offset)

    n_features = data.shape[1]
    print('Initial dataset', data.shape)
    prep_data = pd.DataFrame()
    for company_offset in range(0, len(data), data_size_per_company):
        df_supervised = df_to_supervised(
            data.iloc[company_offset:company_offset+data_size_per_company,:],
            lag,
            no_multisteps,
            vars_output)
        prep_data = pd.concat([prep_data, df_supervised]
            ).reset_index(drop=True)
    print(prep_data.columns)
    print('prep_data', prep_data.shape)

    data_size_per_company = data_size_per_company - lag - no_multisteps + 1
    ahead = config_preprocessing.get('ahead', 1)

    offset = int((1-test_size)*data_size_per_company)
    tst_offset = data_size_per_company - offset
    print('offset', offset)
    print('tst_offset', tst_offset)

    train_set = pd.DataFrame()
    test_set = pd.DataFrame()

    for company_offset in range(0, len(prep_data), data_size_per_company):
        tr = prep_data.iloc[company_offset:offset+company_offset,:]
        tst = prep_data.iloc[offset+company_offset:company_offset+data_size_per_company,:]
        if config_preprocessing.get('stationary'):
            tr = difference(tr.values, tr.columns, ahead)
            tst =  difference(tst.values, tst.columns, ahead)
        train_set = train_set.append(tr)
        test_set = test_set.append(tst)

    scaler, train_scaled, test_scaled = scale(train_set, test_set)
    print('train, test shape', train_scaled.shape, test_scaled.shape)
    ################
    # Model
    modfile = output_dir + '/model.h5'

    # we want to load in each batch the data for 1 company and then reset the state
    batch_size = int(train_scaled.shape[0]/offset)
    tst_batch_size = int(test_scaled.shape[0]/tst_offset)

    train_x, train_y = train_scaled[:, 0:-no_outputs], train_scaled[:, -no_outputs:]
    train_x = train_x.reshape((train_x.shape[0], lag, n_features))

    test_x, test_y = test_scaled[:, 0:-no_outputs], test_scaled[:, -no_outputs:]
    test_x = test_x.reshape((test_x.shape[0], lag, n_features))

    if config['arch']['rnn'] == 'MLP':
        n_input = train_x.shape[1] * train_x.shape[2]
        train_x = train_x.reshape((train_x.shape[0], n_input))
        n_input = test_x.shape[1] * test_x.shape[2]
        test_x = test_x.reshape((test_x.shape[0], n_input))

    model = architecture(neurons=config['arch']['neurons'],
                        drop=config['arch']['drop'],
                        nlayers=config['arch']['nlayers'],
                        activation=config['arch']['activation'],
                        activation_r=config['arch']['activation_r'],
                        rnntype=config['arch']['rnn'], 
                        impl=config['arch']['impl'],
                        train_x=train_x,
                        no_outputs=no_outputs
                        )

    # Training
    optimizer = config['training']['optimizer']

    if optimizer == 'rmsprop':
        if 'lrate' in config['training']:
            optimizer = RMSprop(lr=config['training']['lrate'],
                        decay=config['training']['lrate']//config['training']['epochs'])
        else:
            optimizer = RMSprop(lr=0.001)
    else:
        optimizer = Adam(lr=config['training']['lrate'],
                        decay=config['training']['lrate']//config['training']['epochs'])

    cbacks = []

    tensorboard = TensorBoard(log_dir=output_dir+"/{}".format(time()))
    cbacks.append(tensorboard)

    mcheck = ModelCheckpoint(filepath=modfile, monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto', period=1)
    cbacks.append(mcheck)

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    start = time()
    history = model.fit(train_x, train_y, 
                batch_size=batch_size,
                epochs=config['training']['epochs'],
                validation_data=(test_x, test_y),
                callbacks=cbacks,
                shuffle=False,
                verbose=1)
    train_duration = time() - start

    ############################################
    # Results
    score = model.evaluate(test_x, test_y, batch_size=tst_batch_size, verbose=0)

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    #Loss plot
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig(f'{output_dir}/loss.png')
    plt.close()

    model = load_model(modfile)
    # test batch size equal to the number of company

    train_forecasts = model.predict(train_x, batch_size=batch_size)
    test_forecasts = model.predict(test_x, batch_size=batch_size)

    if no_outputs == 1:
        pass
        #test_forecasts = test_forecasts.reshape(1,-1)
        #train_forecasts = train_forecasts.reshape(1,-1)

    print('Forecasts shape', train_forecasts.shape, test_forecasts.shape)
    train_scaled[:,-no_outputs:] = train_forecasts
    train_forecasts_inv = scaler.inverse_transform(train_scaled)[:,-no_outputs:]

    test_scaled[:,-no_outputs:] = test_forecasts
    test_forecasts_inv = scaler.inverse_transform(test_scaled)[:,-no_outputs:]
    print('Train inv scale forecast', train_forecasts_inv.shape)
    print('Test inv scale forecast', test_forecasts_inv.shape)

    tr_forecast_lst = np.array([])
    tst_forecast_lst = np.array([])
    tr_actual_lst = np.array([])
    tst_actual_lst = np.array([])


    i = 0
    if config_preprocessing.get('stationary'):
        offset = offset - ahead
        tst_offset = tst_offset - ahead
    # inverse difference
    for company_offset in range(0, len(prep_data), data_size_per_company):

        tr = prep_data.iloc[company_offset:offset+company_offset, -no_outputs:].values
        tst = prep_data.iloc[offset+company_offset:company_offset+offset+tst_offset, -no_outputs:].values

        if config_preprocessing.get('stationary'):
            tr_actual_values = prep_data.iloc[company_offset+ahead:offset+company_offset+ahead, -no_outputs:].values
            
            tst_actual_values = prep_data.iloc[offset+company_offset+ahead*2:company_offset+offset+tst_offset+ahead*2, -no_outputs:].values
        else:
            tr_actual_values = prep_data.iloc[company_offset:offset+company_offset, -no_outputs:].values
            
            tst_actual_values = prep_data.iloc[offset+company_offset:company_offset+offset+tst_offset, -no_outputs:].values
        
        tr_actual_lst = np.append(tr_actual_lst, tr_actual_values)
        tr_forecast = train_forecasts_inv[i*offset:offset*(i+1),:]
        
        tst_actual_lst = np.append(tst_actual_lst, tst_actual_values)
        tst_forecast = test_forecasts_inv[i*tst_offset:tst_offset*(i+1),:]

        if config_preprocessing.get('stationary'):
            tr_forecast_lst = np.append(tr_forecast_lst, tr_forecast + tr)
            tst_forecast_lst = np.append(tst_forecast_lst, tst_forecast + tst)
        else:
            tr_forecast_lst = np.append(tr_forecast_lst, tr_forecast)
            tst_forecast_lst = np.append(tst_forecast_lst, tst_forecast)
        i += 1

    # Evaluate
    train_mse = mean_squared_error(tr_actual_lst, tr_forecast_lst)
    test_mse = mean_squared_error(tst_actual_lst, tst_forecast_lst)
    train_mse_persist = mean_squared_error(tr_actual_lst[ahead:], tr_actual_lst[0:-ahead])
    test_mse_persist = mean_squared_error(tst_actual_lst[ahead:], tst_actual_lst[0:-ahead])

    print('Test MSE score', score)
    print('Train MSE', train_mse)
    print('Test MSE', test_mse)
    print('Train MSE persistence', train_mse_persist)
    print('Test MSE persistence', test_mse_persist)

    train_r2 = r2_score(tr_actual_lst, tr_forecast_lst)
    test_r2 = r2_score(tst_actual_lst, tst_forecast_lst)
    train_r2_persist = r2_score(tr_actual_lst[ahead:], tr_actual_lst[0:-ahead])
    test_r2_persist = r2_score(tst_actual_lst[ahead:], tst_actual_lst[0:-ahead])
    print('Train R2', train_r2)
    print('Test R2', test_r2)
    print('Train R2 persistence', train_r2_persist)
    print('Test R2 persistence', test_r2_persist)

    np.save(f'{output_dir}/loss.npy', loss)
    np.save(f'{output_dir}/val_loss.npy', val_loss)
    np.save(f'{output_dir}/tr_actual_lst.npy', tr_actual_lst)
    np.save(f'{output_dir}/tr_forecast_lst.npy', tr_forecast_lst)
    np.save(f'{output_dir}/tst_actual_lst.npy', tst_actual_lst)
    np.save(f'{output_dir}/tst_forecast_lst.npy', tst_forecast_lst)
    
    
    resfile = open('result-rnn.txt', 'a')
    resfile.write('DATAS= %s, LAG= %d, STATIONARY= %s, MULTIVAR= %s, REFIT= %s, MULTISTEPS= %d, RNN= %s, NLAY= %d, NNEUR= %d, DROP= %3.2f, ACT= %s, RACT= %s, OPT= %s, EPOCH= %d OutputDir= %s \nTrainDuration=%3.5f, train_mse=%3.5f, test_mse=%3.5f, train_mse_persist=%3.5f, test_mse_persist=%3.5f mse_score=%3.5f\ntrain_r2=%3.5f, test_r2= %3.5f, train_r2_persist=%3.5f, test_r2_persist=%3.5f \n\n' %
                    (config['data']['dataset'],
                    config['preprocessing']['lag'],
                    str(config['preprocessing']['stationary']),
                    str(config['preprocessing']['multivariate']),
                    str(config['preprocessing']['refit']),
                    config['preprocessing']['no_multisteps'],
                    config['arch']['rnn'],
                    config['arch']['nlayers'],
                    config['arch']['neurons'],
                    config['arch']['drop'],
                    config['arch']['activation'],
                    config['arch']['activation_r'],
                    config['training']['optimizer'],
                    config['training']['epochs'],
                    output_dir,
                    train_duration,
                    train_mse, test_mse, train_mse_persist, test_mse_persist, score,
                    train_r2, test_r2, train_r2_persist, test_r2_persist
                    ))
    resfile.close()
    
    dict_results = {
        'OutDir': output_dir,
        'Lag': lag,
        'Ahead': ahead,
        'Msteps': no_multisteps,
        'StatVar': str(config['preprocessing']['stationary']),
        'Mvar': str(config['preprocessing']['multivariate']),
        'ReFit': str(config['preprocessing']['refit']),
        'Net': config['arch']['rnn'],
        'Nlay': config['arch']['nlayers'],
        'Nneur': config['arch']['neurons'],
        'Opt': config['training']['optimizer'],
        'Drop': round(config['arch']['drop'],3),
        'Epoch': config['training']['epochs'],
        'Activ': config['arch']['activation'],
        'RActiv': config['arch']['activation_r'],
        'TrDur': round(train_duration, 4),
        'MSEscore': round(score, 4),
        'TrMSE': round(train_mse, 4),
        'TstMSE': round(test_mse, 4),
        'TrMSEpers': round(train_mse_persist, 4),
        'TstMSEpers': round(test_mse_persist, 4),
        'TrR2': round(train_r2, 4),
        'TstR2': round(test_r2, 4),
        'TrR2pers': round(train_r2_persist, 4),
        'TstR2pers': round(test_r2_persist, 4)
    }
    df_results = pd.DataFrame([dict_results])
    print(df_results.columns)
    df_results.to_csv('results-rnn.csv', mode='a', header=False)

    return score
