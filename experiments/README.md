
## Notes on experiment reproduction.

The experiment reproduces the settings of Atashgahi et al. (2023) (https://openreview.net/forum?id=GcO6ugrLKp)

Required Steps:
1. Download the Datasets. Download links as well as the exact required file locations within experiments/data/ can be seen from experiments/utils.py
2. Install torchvision (e.g., via `pip install torchvision`).
3. (Optional) Test preprocessing by running `run_preprocessing.py`. This step ensures that the results for all features match the "Baseline" results reported in Atashgahi et al. (2023)
4. Execute run_main.py
