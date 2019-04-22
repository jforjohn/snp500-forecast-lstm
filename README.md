# sp500-forecast-lstm
Explore RNN-type networks for predicting time series sequence for the use case of S&amp;P 500 stock data.

Data available in [S&P 500 stock data](https://www.kaggle.com/camnugent/sandp500) challenge in kaggle. 

Here is a description of what is the purpose of (almost) each file:
- *config.json*: it's an example of all the parameters you can change in the experiments you want to run
- *MyPreprocessing.py*: 
  - it filters the dataset with the sector of companies existing in the data directory
  - it transforms the time series in supervised format (X,y)
  - it transforms the data to a scaled format
- *MyModels.py*: it's a template-like script that creates the MLP or GRU/LSTM RNN network.
- *MainLauncher.py*: it takes as a parameter a json config file
  - reads the dataset
  - call the preprocessing methods
  - creates the train,test datasets
  - fits the model
  - predicts the test set
  - evaluates the model
  - writes the results to files
- *run.py* runs experiments with different lag windows and the best one in terms of Mean Squared error it uses it to run experiments with different number of neurons. Everytime it calls the MainLauncher passing the appropriate config file.