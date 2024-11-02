import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import math

# test
def get_train_test_data(n=100, seed=11121):

    data_all = datasets.make_moons(n, random_state=seed)[0]

    train_n = math.floor(0.8 * n)
    train_data = data_all[:train_n]
    test_data = data_all[train_n:]

    return train_data, test_data


def gaussian_kernel(x, data_points, sigma):

    diff = x - data_points
    d = data_points.shape[1]
    return (1 / ((2 * np.pi) ** (d / 2) * sigma ** d)) * np.exp(-np.sum(diff ** 2, axis=1) / (2 * sigma ** 2))


def kde(data, x_points, sigma):
    kde_values = np.zeros(x_points.shape[0])

    for i, x in enumerate(x_points):
        kde_values[i] = np.mean(gaussian_kernel(x, data, sigma))  # Vectorized kernel calculation

    return kde_values


def sample_kde(data, sigma, n_samples=1):

    points = data[np.random.choice(len(data), n_samples)]
    samples = points + np.random.normal(scale=sigma, size=points.shape)
    return samples


def plot_kde(data, kde_values, x_points, title="KDE for make_moons Data"):
    plt.figure(figsize=(8, 6))

    plt.scatter(x_points[:, 0], x_points[:, 1], c=kde_values, cmap="viridis", s=10, alpha=0.6, label="KDE")
    plt.colorbar(label="Density")
    plt.scatter(data[:, 0], data[:, 1], color='red', s=30, edgecolor="black", label="Training Data")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_data, test_data = get_train_test_data()

    # Step 2: Define Evaluation Points
    x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
    y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    x_points = np.c_[x_grid.ravel(), y_grid.ravel()]  # Flatten the grid points for evaluation

    sigma = 0.3  # Bandwidth parameter, adjust as needed
    kde_values = kde(train_data, x_points, sigma)
    kde_values = kde_values.reshape(x_grid.shape)  # Reshape to match the grid for plotting

    plot_kde(train_data, kde_values, x_points, title="KDE for make_moons Data")

    generated_samples = sample_kde(train_data, sigma, n_samples=100)
    plt.figure()
    plt.scatter(train_data[:, 0], train_data[:, 1], color='blue', s=10, alpha=0.5, label="Training Data")
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], color='purple', s=10, label="Generated Samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Generated Samples from KDE")
    plt.show()
