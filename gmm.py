from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import multivariate_normal

def get_train_test_data(n=500, seed=11121):
    # Generate all data
    data_all = datasets.make_moons(n, random_state=seed)[0]

    # Split into train/test data
    train_n = np.floor(0.8 * n).astype(int)
    train_data = data_all[:train_n]
    test_data = data_all[train_n:]

    return train_data, test_data

def random_initialisation(train_data, C, cov_normalisation=0.2):
    # Weights are uniform
    weights_array = np.ones(C) * 1 / C

    # Select C data points to act as initial means
    data_indices = np.random.choice(len(train_data), C, replace=False)
    means_array = train_data[data_indices]

    # Take empirical covariance and multiply by a normalisation factor for random covariances
    data_empirical_covariance = np.cov(train_data, rowvar=False)
    covariances_array = np.repeat(data_empirical_covariance[np.newaxis], C,
                                  axis=0) * cov_normalisation

    return weights_array, means_array, covariances_array

def kmeans_initialisation(train_data, C, seed):
    # Weights are uniform
    weights_array = np.ones(C) * 1 / C

    # Perform k means clustering; initial means are cluster centers, initial variances are cluster empirical variances
    kmeans = KMeans(n_clusters=C, init='k-means++', random_state=seed).fit(train_data)

    ## Obtain means
    means_array = kmeans.cluster_centers_

    ## Obtain covariance arrays by getting empirical covariance within each
    covariances_array = np.zeros((C, 2, 2))
    for k in range(C):
        cluster_data = train_data[kmeans.labels_ == k]
        deviations = cluster_data - means_array[k]
        covariances_array[k] = np.cov(deviations, rowvar=False)

    return weights_array, means_array, covariances_array

def get_influence_array(means_array, covariances_array, weights_array, train_data, C):
    # Store component and full distributions for getting the influence array
    component_dists = [multivariate_normal(means_array[i], covariances_array[i]) for i in range(C)]

    # define pdf_values: element i,k is the probability density of datapoint i in component k
    pdf_values = np.array([[dist.pdf(data_point) for dist in component_dists] for data_point in train_data])

    # Define gamma array: gamma_i,k = 'influences of component k on instance i'
    influence_array = weights_array * pdf_values / np.sum(weights_array * pdf_values, axis=1, keepdims=True)

    return influence_array

def update_parameters(train_data, influence_array, C):
    n = len(train_data)
    # Update parameters
    weights_array = np.sum(influence_array, axis = 0)/n

    means_array = np.zeros((C, 2))
    for k in range(C):
        means_array[k] = np.sum((influence_array[:, k, np.newaxis]*train_data), axis = 0)/(n*weights_array[k])

    covariances_array = np.zeros((C, 2, 2))
    for k in range(C):
        deviations = train_data - means_array[k]
        covariances_array[k] = \
            np.array([influence_array[i,k]*np.outer(deviations[i], deviations[i]) for i in range(n)]).sum(axis=0) \
            /influence_array.sum(axis = 0)[k]

    return weights_array, means_array, covariances_array

def make_gmm_pdf(weights_array, means_array, covariances_array):
    def gmm_pdf(x):
        component_dists = [multivariate_normal(means_array[i], covariances_array[i]) for i in range(len(means_array))]
        pdf_value = np.sum([weight*dist.pdf(x) for weight, dist in zip(weights_array, component_dists)])
        return pdf_value

    return gmm_pdf


def train_gmm(train_data, C, num_iter, init_method, init_covariance_normalisation, seed):
    # Get initial guesses
    assert init_method in ['random', 'kmeans++'], 'Invalid initialisation method'

    if init_method == 'random':
        weights_array, means_array, covariances_array = random_initialisation(train_data,
                                                                              C,
                                                                              init_covariance_normalisation)
    elif init_method == 'kmeans++':
        weights_array, means_array, covariances_array = kmeans_initialisation(train_data,
                                                                              C,
                                                                              seed)

    # Iterate EM algorithm
    for t in range(num_iter):
        influence_array = get_influence_array(means_array, covariances_array, weights_array, train_data, C)

        weights_array, means_array, covariances_array = update_parameters(train_data, influence_array, C)

    return weights_array, means_array, covariances_array

def sample_from_gmm(weights_array, means_array, covariances_array, n, seed):
    if seed:
        np.random.seed(seed)

    # Choose mixture component
    k = np.random.choice(len(weights_array), p=weights_array)

    chosen_mvn = multivariate_normal(means_array[k], covariances_array[k])
    samples = chosen_mvn.rvs(n)

    return samples



if __name__ == '__main__':
    # Fix data parameters
    n = 50
    init_method = 'random' # or kmeans++
    seed = 11121

    # Fix GMM hyperparameters:
    C = 5
    num_iter = 30
    init_covariance_normalisation = 0.1 # only for random init


    # Get data
    train_data, test_data = get_train_test_data(n, seed)

    weights_array, means_array, covariances_array = \
        train_gmm(train_data, C, num_iter, init_method, init_covariance_normalisation, seed)

    learned_samples = sample_from_gmm(weights_array, means_array, covariances_array, n, seed)

    


