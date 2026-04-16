import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from LinearRegressionModel import Model_LR as model


fig, ax = plt.subplots(2, 3)

dataset = sklearn.datasets.fetch_california_housing().data
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude']

labels = dataset[:, 2] # average rooms

print(dataset)
print(labels[:20])

# Labels - AveRooms

# first model. Using all features
features = np.delete(dataset, 2, axis=1)
print(features[0])