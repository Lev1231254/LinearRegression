import numpy as np
import pandas as pd

class Model_LR:
    def __init__(self, lr, epochs, batch_size, L2_potency):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.L2_potency = L2_potency

        self.weights = None
        self.bias = 0
        self.weights_history = []

        self.losses_history = []
        self.losses_val_history = []
        self.losses_L2_history = []

        # normalization parameters
        self.mean = 0
        self.std = 1


    def normalize(self, features):
        return (features - self.mean) / self.std


    def fit(self, features_train, labels_train, features_val, labels_val):
        self.weights = np.zeros(len(features_train[0]))

        # set normalization parameters
        self.mean = features_train.mean(axis=0)
        self.std = features_train.std(axis=0)

        # normalize features
        features_train = self.normalize(features_train)
        features_val = self.normalize(features_val)


        for epoch in range(self.epochs):

            for i in range(0, len(labels_train), self.batch_size):
                features_train_batch = features_train[i : i + self.batch_size]
                labels_train_batch = labels_train[i : i + self.batch_size]

                self.update_weights(features_train_batch, labels_train_batch)

            self.update_losses(features_train, labels_train, features_val, labels_val)
                
    
    def update_weights(self, features_batch, labels_batch):
        prediction = self.predict(features_batch)

        loss = prediction - labels_batch
        n = len(labels_batch)

        dw = 1 / n * np.dot(features_batch.T, loss) + 2 * self.L2_potency * self.weights
        db = 1 / n * np.sum(loss)

        self.weights -= self.lr * dw
        self.bias -= self.lr * db

        self.weights_history.append(self.weights.copy())


    def predict(self, features):
        return np.dot(features, self.weights) + self.bias
    
    
    def update_losses(self, features, labels, features_val, labels_val):
        # normal history
        prediction = self.predict(features)
        loss = prediction - labels
        
        MSE = np.mean(loss ** 2)
        self.losses_history.append(MSE)

        # L2 history
        if self.L2_potency > 0:
            loss = prediction - labels

            MSE = np.mean(loss ** 2)
            l2 = self.L2_potency * sum(self.weights ** 2)

            total_loss = MSE + l2
            self.losses_L2_history.append(total_loss)


        # validation history
        if len(features_val) > 0:
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
        features_train = self.normalize(features_train)
        ftrs_val = self.normalize(ftrs_val)
        ftrs_test = self.normalize(ftrs_test)
        
        
        # plot predictions
        predictions = self.predict(ftrs_test)
        w, b = np.polyfit(predictions, lbls_test, 1)
        axis[row][0].scatter(predictions, lbls_test)

        axis[row][0].plot([0, max(predictions)], 
                          [b, w * max(predictions) + b], 
                          color = 'red')
        axis[row][0].set_title('Predictions / Actual values')



        # plot weights history
        weights_array = np.array(self.weights_history)  # shape: (steps, n_features)

        for i in range(weights_array.shape[1]):
            axis[row][1].plot(weights_array[:, i], label=f'w{i}')

        axis[row][1].set_title('Weights history')
        axis[row][1].legend()


        # plot losses history
        axis[row][2].plot(self.losses_history, label='Train loss')

        if len(self.losses_val_history) > 0:
            axis[row][2].plot(self.losses_val_history, label='Val loss')

        if len(self.losses_L2_history) > 0:
            axis[row][2].plot(self.losses_L2_history, label='Total loss: MSE + L2')

        axis[row][2].set_title('Loss (MSE)')
        axis[row][2].legend()