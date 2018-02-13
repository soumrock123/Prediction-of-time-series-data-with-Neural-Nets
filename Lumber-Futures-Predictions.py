import pandas
import math
from math import sqrt
import numpy
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt

# fixing random seed for reproducibility, loading the dataset as a csv file and reversing the order of rows chronologically

numpy.random.seed(7)
dataframe = pandas.read_csv(r'C:\Users\Babu@20June\Lumber-futures.csv', usecols = [1,2,3,4,6,7,8], engine = 'python', skipfooter = 3)
dataframe = dataframe.iloc[::-1].reset_index(drop=True)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalizing the dataset

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# splitting into train and test sets and obtaining their transpose matrices

train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
transpose_train = list(map(list, zip(*train)))
transpose_test = list(map(list, zip(*test)))
look_back = 3
test_X=[]
test_Y=[]
train_X=[]
train_Y=[]	

# creating a function that defines the X and Y using a variable look_back window

def create_dataset(dataset, look_back):
	setX, setY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		setX.append(a)
		setY.append(dataset[i + look_back])
	return numpy.array(setX), numpy.array(setY)

# segregating the dataset according to each feature column

for i in range(len(transpose_train)):
    c,d = create_dataset(transpose_train[i], look_back)
    train_X.append(c)
    train_Y.append(d)
for i in range(len(transpose_test)):
    c,d = create_dataset(transpose_test[i], look_back)
    test_X.append(c)
    test_Y.append(d)

# reshaping the data for tensor operations and then defining and fitting a LSTM model

pred_yhat=[]
for i in range(len(train_X)):
    print(train_X[i].shape, train_Y[i].shape, test_X[i].shape, test_Y[i].shape)
    train_X[i] = numpy.reshape(train_X[i], (train_X[i].shape[0],1, train_X[i].shape[1]))
    test_X[i] = numpy.reshape(test_X[i], (test_X[i].shape[0],1, test_X[i].shape[1]))
    model = Sequential()
    model.add(LSTM(128, input_shape=(1, look_back),activation='tanh', use_bias=True))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(train_X[i], train_Y[i], epochs=25, batch_size=5, verbose=2, shuffle=False)

# making a prediction and plotting the results

    columns = ['Open', 'High', 'Low', 'Last', 'Settle', 'Volume', 'Previous Day Open Interest']
    yhat = model.predict(test_X[i])
    plt.plot(test_Y[i])
    plt.plot(yhat)
    plt.xlabel("Size of test data")
    plt.ylabel("Normalized data points")
    plt.legend(['predictions', 'test'], loc='upper left')
    plt.title("Plotting test against predictions for " + columns[i])
    plt.show()
    transpose_yhat=list(map(list, zip(*yhat)))
    if i == 0:
        pred_yhat = transpose_yhat
    else:
        pred_yhat = pred_yhat + transpose_yhat
    print(len(pred_yhat))

# printing the final RMSE and also the test and predicted results
    
    rmse = sqrt(mean_squared_error(test_Y[i], yhat))
    print('Test RMSE: %.3f' % rmse)
transpose_test_Y = list(map(list, zip(*test_Y)))
print(scaler.inverse_transform(transpose_test_Y))

print('*******************************************************')
transpose_pred_yhat=list(map(list, zip(*pred_yhat)))
print(scaler.inverse_transform(transpose_pred_yhat))


