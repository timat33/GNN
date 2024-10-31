from sklearn import datasets
import numpy as np

def get_train_test_data(n=500, seed=11121):
    # Generate all data
    data_all = datasets.make_moons(n, random_state=seed)[0]

    # Split into train/test data
    train_n = np.floor(0.8 * n).astype(int)
    train_data = data_all[:train_n]
    test_data = data_all[train_n:]

    return train_data, test_data
