from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.regularizers import L1L2


def architecture(neurons, drop, nlayers, activation, activation_r, rnntype, impl, train_x, no_outputs):
  """
  RNN architecture

  :return:
  """
  if rnntype == 'LSTM':
    RNN = LSTM
  elif rnntype == 'GRU':
    RNN = GRU

  reg = L1L2(l1=0.1, l2=0.1)
  model = Sequential()
  if nlayers == 1:
      if len(train_x.shape) == 2:
        model.add(Dense(neurons, activation=activation, input_dim=train_x.shape[1]))
      else:  
        model.add(
            RNN(neurons,
            input_shape=(train_x.shape[1],
            train_x.shape[2]),
            implementation=impl,
            recurrent_dropout=drop, activation=activation,
            recurrent_activation=activation_r, 
            stateful=False,
            kernel_regularizer=reg))
  else:
      if len(train_x.shape) == 2:
          for _ in range(1, nlayers - 1):
            model.add(Dense(neurons, activation=activation))  

      else:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl,
                        recurrent_dropout=drop, activation=activation, recurrent_activation=activation_r,
                        return_sequences=True))
        for _ in range(1, nlayers - 1):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl,
                            activation=activation, recurrent_activation=activation_r, return_sequences=True))
        model.add(RNN(neurons, recurrent_dropout=drop, activation=activation,
                        recurrent_activation=activation_r, implementation=impl))
  
  model.add(Dense(no_outputs))
  print(model.summary())

  return model
