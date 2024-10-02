 
import relchanet

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import preprocessing  
from sklearn.svm import SVC

# Generate a synthetic dataset with 100 features
X, y = make_classification(n_samples=10000, n_features=100, n_informative=10, n_redundant=10, n_classes=2, random_state=42,shuffle = False)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data:
X = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(np.concatenate((X_train, X_test)))
X_train = X[: y_train.shape[0]]
X_test = X[y_train.shape[0]:]
        
# Set Parameters
args = {
# General Parameters:
    'epochs': 2000,
    'lr': 1e-3,
    'batch_size': 1024,
    'n_hidden': 10, # First Layer size
# Specific Parameters: 
    'switch_interval': 100, # Controls number of mini-batches over which relative score is calculated
    'candidate_perc': .2, # Controls input layer size
}

# Set number of features to be selected
K = 10 

# Select features
mod = relchanet.rcn_class(X=X_train, y=y_train, relevant=K, **args)

# get selected features
selected_features = mod.ident_final
print(selected_features) # Selected Features, (0-9 are actually informative,10-19 are redundant)

# train downstream learner
learner = SVC()
learner.fit(X_train[:, selected_features], y_train)

# Test the learner on the test data
y_pred = learner.predict(X_test[:, selected_features])

# Compute the accuracy
accuracy = accuracy_score(y_test, y_pred)          
print(f"Test Accuracy: {accuracy:.4f}")

