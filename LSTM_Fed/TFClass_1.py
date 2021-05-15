import tensorflow as tf
import csv
import numpy as np
import os
import warnings
import random
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
import math

warnings.filterwarnings("ignore")

class TFML:
    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def dataProcess(self,look_back):
        dataframe = read_csv('BTC Dataset (with Sentiment).csv', usecols=[4], engine='python')
        dataset = dataframe.values
        dataset = dataset.astype('float32')

        # normalize the dataset

        dataset = self.scaler.fit_transform(dataset)
        # split into train
        train_size = int(len(dataset) * 0.9)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
        # reshape into X=t and Y=t+1
        #look_back = 1
        trainX, trainY = self.create_dataset(train, look_back)

        testX, testY = self.create_dataset(test, look_back)
        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        return trainX, trainY, testX, testY,dataset

    def __init__(self,name):
        self.name=name
        self.look_back=1
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.trainX,self.trainY,self.testX, self.testY,self.dataset=self.dataProcess(self.look_back)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(4, input_shape=(1, self.look_back)),
            tf.keras.layers.Dense(1, activation='relu')
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')


    def run(self):
        self.model.fit(self.trainX, self.trainY, epochs=1, batch_size=25, validation_data=(self.testX, self.testY),verbose=2)

    def eval(self):
        #trainPredict = self.model.predict(self.trainX)
        testPredict = self.model.predict(self.testX)
        # invert predictions
        #trainPredict = self.scaler.inverse_transform(trainPredict)
        #trainY = self.scaler.inverse_transform([self.trainY])
        testPredict = self.scaler.inverse_transform(testPredict)
        ttestY = self.scaler.inverse_transform([self.testY])
        # calculate root mean squared error
        #trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        #print('Train Score: %.2f RMSE' % (trainScore))
        #testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        testScore=mean_absolute_percentage_error(ttestY[0], testPredict[:, 0])
        print('Accuracy: '+str(1-testScore))
        #print('Test Score: %.2f RMSE' % (testScore))
        # shift train predictions for plotting
        #trainPredictPlot = np.empty_like(self.dataset)
        #trainPredictPlot[:, :] = np.nan
        #trainPredictPlot[self.look_back:len(trainPredict) + self.look_back, :] = trainPredict
        #print(trainPredictPlot)
        # shift test predictions for plotting
        #testPredictPlot = np.empty_like(self.dataset)
        #testPredictPlot[:, :] = np.nan
        #testPredictPlot[len(trainPredict) + (self.look_back * 2) + 1:len(self.dataset) - 1, :] = testPredict
        # plot baseline and predictions
        #plt.plot(self.scaler.inverse_transform(self.dataset)[len(trainPredict) + self.look_back + 1:])

        # plt.plot(trainPredictPlot)
        #plt.plot(testPredict)
        #plt.show()











