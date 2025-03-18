from sklearn.datasets import fetch_california_housing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Use only a subset of the data to reduce memory usage
subset_size = 2000  # Adjust this to fit your system's memory
X_subset = X[:subset_size]
y_subset = y[:subset_size]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.3, random_state=42)

# Define a simpler kernel to reduce memory usage
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)

# Create Gaussian Process Regressor
gp = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=10, random_state=42)

# Fit the model
gp.fit(X_train, y_train)

# Make predictions
y_pred = gp.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
