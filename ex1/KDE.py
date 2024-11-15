import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import math
from MMD.MMD import calculate_mmd
import seaborn as sns

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

def Calculate_MMD(bandwidths, n_samples):
    test_data_n = []
    train_data_n = []
    for n in n_samples:
        train, test = get_train_test_data(n)
        test_data_n.append(test)
        train_data_n.append(train)

    mmd_values_exponential = np.zeros((len(train_data_n), len(bandwidth)))
    mmd_values_multiquadratic = np.zeros((len(train_data_n), len(bandwidth)))

    for j, train_j in enumerate(train_data_n):
        for i, width in enumerate(bandwidth):

            n = len(train_j)
            # Step 2: Define Evaluation Points
            x_min, x_max = train_j[:, 0].min() - 1, train_j[:, 0].max() + 1
            y_min, y_max = train_j[:, 1].min() - 1, train_j[:, 1].max() + 1
            x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            x_points = np.c_[x_grid.ravel(), y_grid.ravel()]  # Flatten the grid points for evaluation

            sigma = width  # Bandwidth parameter
            kde_values = kde(train_j, x_points, sigma)
            kde_values = kde_values.reshape(x_grid.shape)

            generated_samples = sample_kde(train_j, sigma, n_samples=n)

            mmd_score = calculate_mmd(train_j, generated_samples, bandwidths=[1.0, 2.0])
            for kernel, mmd_value in mmd_score.items():
                if kernel == 'squared_exponential':
                    mmd_values_exponential[j, i] = mmd_value

                else:
                    mmd_values_multiquadratic[j, i] = mmd_value

    return mmd_values_exponential,mmd_values_multiquadratic

def Plot_MMD_SampleSize_Bandwidth(data, bandwidth, N, Text):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data, annot=True, fmt=".6f", cmap="coolwarm",
                xticklabels=np.round(bandwidth, 2),
                yticklabels=N,
                vmin=0, vmax=0.005)
    plt.xticks(range(4), bandwidth)
    plt.title(Text)
    plt.xlabel('Kernel Bandwidth')
    plt.ylabel('Training Set Size')
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

    #Evaluate MMD
    bandwidth = [0.001,0.01,0.1,0.5]
    n_samples = [100,200,500]

    mmd_values_exponential,mmd_values_multiquadratic = Calculate_MMD(bandwidth, n_samples)
    Plot_MMD_SampleSize_Bandwidth(mmd_values_exponential,bandwidth, n_samples, 'Squared Exponential MMD')
    Plot_MMD_SampleSize_Bandwidth(mmd_values_multiquadratic,bandwidth, n_samples, 'Inverse Multi-Quadratic MMD')

    #plot some good and bad examples
    #IMQ
    #good: bandwidth = 0.01, data size = 200

    train, test = get_train_test_data(200)

    # Step 2: Define Evaluation Points
    x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
    y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    x_points = np.c_[x_grid.ravel(), y_grid.ravel()]  # Flatten the grid points for evaluation

    sigma = 0.01  # Bandwidth parameter
    kde_values = kde(train_data, x_points, sigma)
    kde_values = kde_values.reshape(x_grid.shape)

    plot_kde(train_data, kde_values, x_points, title="Good IMQ: N= 200, h = 0.01")

    generated_samples = sample_kde(train_data, sigma, n_samples=200)

    plt.figure()
    plt.scatter(train_data[:, 0], train_data[:, 1], color='blue', s=10, alpha=0.5, label="Training Data")
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], color='purple', s=10, label="Generated Samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Good IMQ: N= 200, h = 0.01")
    plt.show()


    #bad:  bandwidth = 0.5, data size = 100
    train, test = get_train_test_data(100)

    # Step 2: Define Evaluation Points
    x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
    y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    x_points = np.c_[x_grid.ravel(), y_grid.ravel()]  # Flatten the grid points for evaluation

    sigma = 0.5  # Bandwidth parameter
    kde_values = kde(train_data, x_points, sigma)
    kde_values = kde_values.reshape(x_grid.shape)

    plot_kde(train_data, kde_values, x_points, title="Bad IMQ: N= 100, h = 0.5")

    generated_samples = sample_kde(train_data, sigma, n_samples=100)

    plt.figure()
    plt.scatter(train_data[:, 0], train_data[:, 1], color='blue', s=10, alpha=0.5, label="Training Data")
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], color='purple', s=10, label="Generated Samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Bad IMQ: N= 100, h = 0.5")
    plt.show()



    #SE
    # good: bandwidth = 0.01, data size = 200

    train, test = get_train_test_data(200)
    # Step 2: Define Evaluation Points
    x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
    y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    x_points = np.c_[x_grid.ravel(), y_grid.ravel()]  # Flatten the grid points for evaluation

    sigma = 0.01  # Bandwidth parameter
    kde_values = kde(train_data, x_points, sigma)
    kde_values = kde_values.reshape(x_grid.shape)

    plot_kde(train_data, kde_values, x_points, title="Good SE: N= 200, h = 0.01")

    generated_samples = sample_kde(train_data, sigma, n_samples=200)

    plt.figure()
    plt.scatter(train_data[:, 0], train_data[:, 1], color='blue', s=10, alpha=0.5, label="Training Data")
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], color='purple', s=10, label="Generated Samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Good SE: N= 200, h = 0.01")
    plt.show()

    # Bad:  bandwidth = 0.1, data size = 100
    train, test = get_train_test_data(100)
    # Step 2: Define Evaluation Points
    x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
    y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    x_points = np.c_[x_grid.ravel(), y_grid.ravel()]  # Flatten the grid points for evaluation

    sigma = 0.1  # Bandwidth parameter
    kde_values = kde(train_data, x_points, sigma)
    kde_values = kde_values.reshape(x_grid.shape)

    plot_kde(train_data, kde_values, x_points, title="Bad SE: N= 100, h = 0.1")

    generated_samples = sample_kde(train_data, sigma, n_samples=100)

    plt.figure()
    plt.scatter(train_data[:, 0], train_data[:, 1], color='blue', s=10, alpha=0.5, label="Training Data")
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], color='purple', s=10, label="Generated Samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Bad SE: N= 100, h = 0.1")
    plt.show()


'''
Observings and interpretation
- we can see from using MMD that different bandwidths make a big difference 
- for the chosen data set sizes, lower bandwidths perform better
- for the different sizes of the datasets, different bandwidths are needed to perform the best
- looking at probability distribution we can see that for the best models the 2 moons are captured very well
- looking at the worst model the shape is not really visible so the distribution is very big and smooth
-> suggesting that bandwidth is to big for the size of the training set

Advantages:
- KDE can be extended to higher dimensions
- when visualizing KDE you can see easily if the sampled data follows the distribution
-> easy for interpretation
-> adjusting by trying different bandwidths
Disadvanatges:
- we can see from implementing that model is depending on bandwidth 
-> choosing right bandwidth can take some time (trial and error or some other methods)
- using higher dimensions, computation can take some time
'''