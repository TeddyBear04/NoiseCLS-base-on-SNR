import numpy as np
import antropy as ant
import nolds
import neurokit2 as nk
# Sample random parameters
def sample_random_parameters(param_grid):
    return {key: np.random.choice(values) for key, values in param_grid.items()}
# Autoregressive Predictions
def autoregressive_predict(model, X_test, steps):
    predictions = []
    test_data = X_test.values
    for step in range(steps):
        sample = test_data[step].reshape(1, -1)
        pred = model.predict(sample)[0]
        predictions.append(pred)
        if step + 1 < len(test_data):
            test_data[step + 1][:-1] = test_data[step][1:]
            test_data[step + 1][-1] = pred
    return predictions
def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError
#fallback if no embedding was ever defined within function or as input
def delay_embed(data, embedding_dimension=20, time_delay=1):
    num_samples = len(data) - time_delay * (embedding_dimension - 1)
    return np.array([data[dim * time_delay:num_samples + dim * time_delay] for dim in range(embedding_dimension)]).T[:,::-1]
# Calculate Second Derivative Variance
def hessian(x):
    return np.gradient(np.gradient(x, axis=0), axis=0)
def calculate_variance_2nd_derivative(embedded_matrix):
    """Variance of second derivatives across the embedded manifold."""
    second_derivatives = np.gradient(np.gradient(embedded_matrix, axis=0), axis=0)
    summed_squared = np.sqrt(np.sum(np.square(second_derivatives), axis=1))
    return np.var(summed_squared)
def calculate_point_errors(y_true, y_pred):
    absolute_error = np.abs(y_true - y_pred)
    squared_error = np.square(y_true - y_pred)
    relative_error = np.abs((y_true - y_pred) / y_true)
    relative_error = np.where(np.isnan(relative_error), 0, relative_error)  # Handle division by zero
    return absolute_error, squared_error, relative_error
# Calculate Complexity Metrics
def calculate_hurst(time_series):
    """Hurst exponent via nolds (vectorized, ~50x faster than custom polyfit loop)."""
    try:
        return nolds.hurst_rs(time_series, fit="poly")  # poly is faster than default ransac
    except Exception:
        return np.nan
def calculate_fisher_information(time_series):
    diff_series = np.diff(time_series)
    var_diff = np.var(diff_series)
    fisher_info = 1 / var_diff if var_diff != 0 else 0
    return fisher_info
def calculate_variance_1st_derivative(embedded_matrix):
    """Variance of first derivatives across the embedded manifold."""
    first_derivatives = np.gradient(embedded_matrix, axis=0)
    summed_squared = np.sqrt(np.sum(np.square(first_derivatives), axis=1))
    return np.var(summed_squared)
def calculate_standard_deviation(time_series):
    return np.std(time_series)
def calculate_mean_absolute_deviation(time_series):
    return np.mean(np.abs(time_series - np.mean(time_series)))
# Function to generate pink noise
def generate_pink_noise(size, level):
    num_sources = 16
    random_sources = np.random.normal(0, level, (num_sources, size))
    pink_noise = np.zeros(size)
    max_shifts = 2 ** num_sources
    index = np.zeros(num_sources, dtype=int)
    for i in range(size):
        pink_noise[i] = np.sum([random_sources[k, index[k]] for k in range(num_sources)])
        # Update indices
        j = num_sources - 1
        while True:
            index[j] += 1
            if index[j] < max_shifts // (2 ** (j + 1)):
                break
            index[j] = 0
            j -= 1
            if j < 0:
                break
    return pink_noise / num_sources
def compute_singular_values(time_series, embedding_dimension=10, time_delay=1):
    """Return singular values of delay-embedded trajectory matrix."""
    A = delay_embed(time_series, embedding_dimension, time_delay)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return s
def relative_decay_singular_values(embedded_matrix):
    """Mean ratio Ïƒ_i / Ïƒ_{i+1} of singular values."""
    U, s, Vt = np.linalg.svd(embedded_matrix, full_matrices=False)
    if len(s) < 2:
        return np.nan
    ratios = s[:-1] / (s[1:] + 1e-12)
    return np.mean(ratios)
def svd_energy(embedded_matrix, k=3):
    """Energy captured by first k singular values of the embedded manifold."""
    U, s, Vt = np.linalg.svd(embedded_matrix, full_matrices=False)
    total_energy = np.sum(s**2)
    return np.sum(s[:k]**2) / total_energy if total_energy > 0 else np.nan
def condition_number(embedded_matrix, eps=1e-6):
    """Condition number = max(Ïƒ)/min(Ïƒ) for embedded manifold."""
    U, s, Vt = np.linalg.svd(embedded_matrix, full_matrices=False)
    s = s[s > eps]
    return np.max(s) / np.min(s) if len(s) > 0 else np.nan
def coefficient_of_variation(embedded_matrix):
    """Coefficient of variation of singular values."""
    U, s, Vt = np.linalg.svd(embedded_matrix, full_matrices=False)
    return np.std(s) / np.mean(s) if np.mean(s) != 0 else np.nan
def spectral_skewness(embedded_matrix):
    """Skewness of singular value spectrum."""
    U, s, Vt = np.linalg.svd(embedded_matrix, full_matrices=False)
    mu, sigma = np.mean(s), np.std(s)
    return np.mean(((s - mu) / sigma) ** 3) if sigma != 0 else np.nan
def permutation_entropy_metric(time_series, dimension=3, delay=1): #wrapper
    """Permutation Entropy via antropy (C-optimized)."""
    return ant.perm_entropy(time_series, order=dimension, delay=delay, normalize=True)
def sample_entropy_metric(time_series):
    """Sample Entropy (SampEn)."""
    try:
        return ant.sample_entropy(time_series)
    except Exception:
        return np.nan
def lempel_ziv_complexity_metric(time_series):
    """Lempel-Ziv Complexity (binary symbolic compressibility)."""
    try:
        return ant.lziv_complexity(time_series)
    except Exception:
        return np.nan
def lyapunov_exponent_metric(time_series):
    """Largest Lyapunov exponent (Î»max)."""
    try:
        return nolds.lyap_r(time_series)
    except Exception:
        return np.nan
