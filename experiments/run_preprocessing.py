# Test whether the preprocessing leads to the same baseline learner performance as in Atashgahi.

# needs to be run with "experiments" as working directory 
# results are saved in "experiments/results/preprocessing.csv"

from utils import load_data
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
import csv


datasets = ['coil20', 'MNIST', 'Fashion-MNIST','USPS', 'isolet', 'har', 'BASEHOCK', 'Prostate_GE', 'arcene', 'SMK', 'GLA-BRA-180']
learners = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(n_neighbors = 1, algorithm = 'brute', n_jobs = 1),
    'ET' : ExtraTreesClassifier(n_estimators = 50, n_jobs = -1),
}

results = []

# set seed
np.random.seed(42)


# Calculate baseline accuracies
# SVC and KNN are deterministic, ET has randomness 
for dataset in datasets: 
    (X_train, y_train), (X_test, y_test) = load_data(dataset)
    for learner_name, learner in learners.items():
        if learner_name == "ET":
            accuracies = []
            for _ in range(5):  # Run 5 repetitions
                learner.fit(X_train, y_train)
                y_pred = learner.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)*100
                accuracies.append(accuracy)
            mean_accuracy = np.mean(accuracies)
            sd_accuracy = np.std(accuracies)
            results.append([dataset, learner_name, mean_accuracy, sd_accuracy])
        else: 
            learner.fit(X_train, y_train)
            y_pred = learner.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)*100
            results.append([dataset, learner_name, accuracy, 0])



# From Atashgahi et al, Table 6 on p.19
svm_accuracies = [100.0, 97.92, 88.3, 97.58, 96.03, 95.05, 91.98, 80.95, 77.5, 86.84, 72.22]
knn_accuracies = [100.0, 96.91, 84.96, 97.37, 88.14, 87.85, 78.7, 76.19, 92.5, 73.68, 69.44]
et_accuracies = [100.0, 96.9, 87.39, 96.51, 94.04, 93.59, 96.99, 85.71, 82.5, 78.95, 69.44]

original_results = []
for i, dataset in enumerate(datasets):
    original_results.append([dataset, 'SVM', svm_accuracies[i]])
    original_results.append([dataset, 'KNN', knn_accuracies[i]])
    original_results.append([dataset, 'ET', et_accuracies[i]])
original_results_dict = {(dataset, learner_name): accuracy for dataset, learner_name, accuracy in original_results}

os.makedirs('results', exist_ok=True)


# Compare the results
with open('results/preprocessing.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Write the header
    writer.writerow(['Dataset', 'Learner', 'Mean Accuracy', 'SD', 'Accuracy Difference'])
    
    for result in results:
        dataset, learner_name, mean_accuracy, sd_accuracy = result
        mean_accuracy = np.round(mean_accuracy, 2)
        original_accuracy = original_results_dict[(dataset, learner_name)]
        accuracy_difference = original_accuracy - mean_accuracy
        row = [dataset, learner_name, mean_accuracy, sd_accuracy, accuracy_difference]
        writer.writerow(row)
