# needs to be run with "experiments" as working directory 
# results are saved in "experiments/results/main.csv"

from utils import load_data
from entryprune_models import *

import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score
import time
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import csv
import os

os.makedirs('results', exist_ok=True)

datasets = ['coil20', 'MNIST', 'Fashion-MNIST','USPS', 'isolet', 'har', 'BASEHOCK', 'Prostate_GE', 'arcene', 'SMK', 'GLA-BRA-180',"CIFAR-10","CIFAR-100"]


methods_wide = {
    'EP_wide': fit_wrapper(stopping = "validation", stopping_hyperpar = 100, perc = .5,switch = 5),
    'flex_wide': fit_flex_wrapper(perc = .5,switch = 5),
}
methods_long = {
    'EP_long': fit_wrapper(stopping = "validation", stopping_hyperpar = 100, perc = .2,switch = 100),
    'flex_long': fit_flex_wrapper(perc = .2,switch = 100),
}

# Downstream learners
learners = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(n_neighbors = 1, algorithm = 'brute', n_jobs = 1),
    'ET' : ExtraTreesClassifier(n_estimators = 50, n_jobs = -1),
}

# Define n_vars_selected
n_vars_selected = [25,50,75,100]

n_each = 5

# For each dataset
for dataset_name in datasets:

    # Load the dataset
    (X_train, y_train), (X_test, y_test) = load_data(dataset_name)

    if X_train.shape[0] < X_train.shape[1]:
        methods = methods_wide
    else:
        methods = methods_long

    # For each method
    for method_name, method in methods.items():
        # For each n_vars_selected
        for n_vars in n_vars_selected:

            for i in range(n_each):

                # set seed
                seed = n_vars*10**4+i
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                if n_vars == 50:
                    learners_to_do = ["SVM","KNN","ET"]
                else: 
                    learners_to_do = ["SVM"]

                filtered_learners = {k: learners[k] for k in learners_to_do}

                # Apply the method to select n_vars variables
                start_time = time.time()
                selected_vars = method(X_train, y_train, n_vars)
                elapsed_time = time.time() - start_time

                # For each downstream learner
                for learner_name, learner in filtered_learners.items():
                    # Train the learner on the selected variables
                    learner.fit(X_train[:, selected_vars], y_train)
                    
                    # Test the learner on the test data
                    y_pred = learner.predict(X_test[:, selected_vars])
                    
                    # Compute the accuracy
                    accuracy = accuracy_score(y_test, y_pred)


                    # Save the results
                    with open('results/main.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([dataset_name, method_name, n_vars, learner_name, accuracy, elapsed_time, i, seed])


