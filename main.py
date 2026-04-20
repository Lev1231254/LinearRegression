import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from LinearRegressionModel import Model_LR as model


fig, ax = plt.subplots(2, 3)
fig.set_size_inches((13, 9))

# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude']
dataset = sklearn.datasets.fetch_california_housing().data
labels = dataset[:, 2] # average rooms

# Labels - AveRooms

learning_rate = 0.0001
epochs = 10000
batch_size = 256

# first model. Using all features

# val_point = 15000
# test_point = 18000
# features = np.delete(dataset, 2, axis=1)

# ftrs_train = features[:val_point]
# lbls_train = labels[:val_point]
# ftrs_val = features[val_point : test_point]
# lbls_val = labels[val_point : test_point]
# ftrs_test = features[test_point:]
# lbls_test = labels[test_point:]


# model_all_features = model(learning_rate, epochs, batch_size, 0)
# model_all_features.fit(ftrs_train, lbls_train, ftrs_val, lbls_val)
# model_all_features.paint_data(ftrs_train, lbls_train, ftrs_val, lbls_val, ftrs_test, lbls_test,
#                               fig, ax, 0)


# second model, using all features, but L2 is on

val_point = 15000
test_point = 18000
features = np.delete(dataset, 2, axis=1)

ftrs_train = features[:val_point]
lbls_train = labels[:val_point]
ftrs_val = features[val_point : test_point]
lbls_val = labels[val_point : test_point]
ftrs_test = features[test_point:]
lbls_test = labels[test_point:]


model_all_features = model(learning_rate, epochs, batch_size, 0.1)
model_all_features.fit(ftrs_train, lbls_train, ftrs_val, lbls_val)
model_all_features.paint_data(ftrs_train, lbls_train, ftrs_val, lbls_val, ftrs_test, lbls_test,
                              fig, ax, 1)
plt.show()