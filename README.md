# Prediction of time series data with Neural Nets
### Predicting how the prices of Lumber-Futures will change in the future

In this project we have considered the problem of predicting the changes in prices of real-time data, that appears in the form of a time-series. We have used neural networks for implementing this task (LSTM to be precise) and our implementing language is Python.

Although the code is self-explanatory, I am just adding a brief overview of the implementation process.
The dataset is in the form of a ".csv" file and has a little above 11000 data samples with 8 features. These features basically represent the opening, closing and highest values of the stock prices for any particular day, along with some other feature parameters. 

Firstly the dataset is rearranged in chronological order and also the data points are normalized. After that it is segregated into train and test sets. We have introduced a lookback window which takes into account a small set of data to predict the next output. This is done so that the trend is not missed. Since there are multiple feature vectors, we have reorganized the data such that we apply the lookback method upon each column vector and predict the output. After that both the train and test datasets are reshaped for tensor operations and next a LSTM model is fitted. Finally we calculate the loss and visualize the test and predicted values using python plots, one for each feature.
