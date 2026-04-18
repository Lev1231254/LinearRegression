import numpy as np
import pandas as pd

class Model_LR:
    def __init__(self, lr, epochs, batch_size):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.weights = None
        self.bias = 0
        self.weights_history = None

        self.losses_history = None
        self.losses_val_history = None


    def fit(self, features_train, labels_train, features_val, labels_val):
        self.weights = np.zeros(len(features_train[0]))
        self.weights_history = []
        self.losses_history = []
        self.losses_val_history = []

        for epoch in range(self.epochs):

            for i in range(0, len(labels_train), self.batch_size):
                features_train_batch = features_train[i : i + self.batch_size]
                labels_train_batch = labels_train[i : i + self.batch_size]

                self.update_weights(features_train_batch, labels_train_batch)

                self.update_losses(features_train_batch, labels_train_batch, features_val, labels_val)
                
    
    def update_weights(self, features_batch, labels_batch):
        prediction = self.predict(features_batch)
        loss = prediction - labels_batch
        n = len(labels_batch)

        dw = 1 / n * np.dot(features_batch.T, loss)
        db = 1 / n * np.sum(loss)

        self.weights -= self.lr * dw
        self.bias -= self.lr * db

        self.weights_history.append(self.weights)


    def predict(self, features):
        return np.dot(features, self.weights) + self.bias
    
    
    def update_losses(self, features, labels, features_val, labels_val):
        # normal history
        prediction = self.predict(features)
        loss = prediction - labels
        
        MSE = np.mean(loss ** 2)
        self.losses_history.append(MSE)


        # validation historyprediction = self.predict(features, labels)
        if (features_val.all() != None and labels_val.all() != None):
            prediction = self.predict(features_val)
            loss = prediction - labels_val

            MSE = np.mean(loss ** 2)
            self.losses_val_history.append(MSE)



    def test_model(self, features_test, labels_test):
        predictions = self.predict(features_test)
        loss = predictions - labels_test

        MSE = np.mean(loss ** 2)
        return MSE
        

    def paint_data(self, features_train, labels_train, ftrs_val, lbls_val, ftrs_test, lbls_test, fig, axis, row):
        predictions = self.predict(ftrs_test)

        w, b = np.polyfit(predictions, lbls_test, 1)
        axis[row][0].scatter(predictions, lbls_test)
        axis[row][0].plot([0, b], [w * len(lbls_test) + b], color = 'red')