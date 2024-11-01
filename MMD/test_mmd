import numpy as np
from MMD import calculate_mmd


def test_mmd_same_distribution():
    """Test MMD on the same distribution. Expected MMD is 0 or close to 0 due to floating-point precision."""
    np.random.seed(0)
    X = np.random.normal(0, 1, (100, 2))
    mmd_values = calculate_mmd(X, X, bandwidths=[1.0, 2.0])

    for kernel, mmd_value in mmd_values.items():
        assert np.isclose(mmd_value, 0, atol=1e-6), f"Failed for {kernel} with MMD={mmd_value}"
    print("test_mmd_same_distribution passed.")


def test_mmd_similar_distributions():
    """Test MMD on similar but not identical distributions. Expected MMD is small but non-zero."""
    np.random.seed(0)
    X_true = np.random.normal(0, 1, (100, 2))
    X_pred = X_true + np.random.normal(0, 0.1, (100, 2))  # Add small noise of normal distr.

    mmd_values = calculate_mmd(X_true, X_pred, bandwidths=[1.0, 2.0])

    for kernel, mmd_value in mmd_values.items():
        assert mmd_value > 0, f"MMD should be greater than 0 for similar distributions with {kernel}"
        print(f"test_mmd_similar_distributions passed for kernel {kernel} with MMD={mmd_value}")


def test_mmd_different_distributions():
    """Test MMD on different distributions. Expected MMD should be significantly greater than zero."""
    np.random.seed(0)
    X_true = np.random.normal(0, 1, (100, 2))
    X_pred = np.random.normal(5, 1, (100, 2))  # Shifted distribution

    mmd_values = calculate_mmd(X_true, X_pred, bandwidths=[1.0, 2.0])

    for kernel, mmd_value in mmd_values.items():
        if kernel == "squared_exponential":
            assert mmd_value > 1, f"MMD should be significantly greater than 1 for different distributions with {kernel}"
        elif kernel == "inverse_multiquadratic":
            assert mmd_value > 0.1, f"MMD should be greater than 0.1 for different distributions with {kernel}"

        print(f"test_mmd_different_distributions passed for kernel {kernel} with MMD={mmd_value}")


if __name__ == "__main__":
    test_mmd_same_distribution()
    test_mmd_similar_distributions()
    test_mmd_different_distributions()
