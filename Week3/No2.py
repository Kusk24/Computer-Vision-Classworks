# Name - Win Yu Maung
# ID - 6612054
# Sec - 541

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generate Synthetic Data ---
np.random.seed(42)  # For reproducibility
num_points = 50

# True line: y = 2x + 1
x_true = np.linspace(0, 10, num_points)
y_true = 2 * x_true + 1

# Add Gaussian noise
noise = np.random.normal(0, 2, num_points)
y_noisy = y_true + noise

# Add 5 significant outliers
outlier_indices = [10, 15, 30, 35, 40]
for idx in outlier_indices:
    y_noisy[idx] += np.random.choice([-15, 15])  # Add a large offset

# --- 2. Least Squares Fitting ---
# Build the design matrix A (N x 2) where each row is [x, 1]
A = np.vstack([x_true, np.ones(len(x_true))]).T
# Solve for the slope (m) and intercept (c) using least squares
m_ls, c_ls = np.linalg.lstsq(A, y_noisy, rcond=None)[0]

print(f"Least Squares Fit: y = {m_ls:.3f}x + {c_ls:.3f}")

# --- 3. Visualization ---
plt.figure(figsize=(10, 6))
plt.scatter(x_true, y_noisy, label='Data with Outliers', color='blue', alpha=0.7)
plt.plot(x_true, y_true, 'g--', label='True Line', linewidth=2)

# Least squares line
plt.plot(x_true, m_ls*x_true + c_ls, 'r-', label='Least Squares Fit', linewidth=2)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Least Squares Fitting (Sensitive to Outliers)')
plt.legend()
plt.grid(True)
plt.show()
