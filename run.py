import json
import sys
from os import mkdir
from MainLauncher import MainLauncher

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

config = load_config_file('config', './')
config_data = config['data']

# Experiments
output_dir_pref = config_data['output_dir'] + '_mlp_nostat_mstep%s' %(config['preprocessing']['no_multisteps'])


results_lag = []
lag_lst = [42, 52,72] #[1,6,14,31]
for lag in lag_lst:
  output_dir = output_dir_pref + '_lag_%s' %(lag)
  try:
      # Create target Directory
      mkdir(output_dir)
  except FileExistsError:
      print("Directory " , output_dir ,  " already exists")
  config['data']['output_dir'] = output_dir
  config['preprocessing']['lag'] = lag
  output_file = output_dir + '/output.txt'
  print(output_file)
  sys.stdout = open(output_file, 'w')
  results_lag.append(MainLauncher(config))

#manual
lag_lst = [31,42,52,72]
####
lag = lag_lst[results_lag.index(min(results_lag))]
'''
lag = 31
config['preprocessing']['lag'] = lag

results_neuron = []
neurons_lst = [40] #[5,15,40,120]
for neuron in neurons_lst:
  output_dir = output_dir_pref + f'_lag_{lag}' + f'_neuron_{neuron}'
  try:
      # Create target Directory
      mkdir(output_dir)
  except FileExistsError:
      print("Directory " , output_dir ,  " already exists")
  config['data']['output_dir'] = output_dir
  config['arch']['neurons'] = neuron
  print(config)
  output_file = output_dir + '/output.txt'
  print(output_file)
  sys.stdout = open(output_file, 'w')
  results_neuron.append(MainLauncher(config))
'''
#manual
neurons_lst = [120, 180, 200]
#######
neuron = neurons_lst[results_neuron.index(min(results_neuron))]
config['arch']['neurons'] = neuron
print('Best neuron', neuron)
