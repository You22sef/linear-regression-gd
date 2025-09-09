# main.py
# Linear Regression: Closed-form (Normal Equation) vs Gradient Descent
# Student: <Yousef Aldandani>
# ID: <700783505>

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Generate dataset
# ---------------------------
np.random.seed(42)                 # make results repeatable
m = 200                            # number of samples
x = np.random.uniform(0, 5, m)     # x in [0,5]
epsilon = np.random.randn(m)       # Gaussian noise ~ N(0,1)
y = 3 + 4 * x + epsilon            # true model y = 3 + 4x + noise

# Plot raw data
plt.figure(figsize=(6,4))
plt.scatter(x, y, alpha=0.7, label='Raw data')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Raw data (y = 3 + 4x + noise)")
plt.legend()
plt.tight_layout()
plt.savefig("raw_data.png")
# plt.show()  # optional

# ---------------------------
# 2. Closed-form solution (Normal Equation)
# ---------------------------
# Add bias column (column of 1s)
X = np.column_stack((np.ones(m), x))   # shape (m, 2)

# Normal equation: theta = (X^T X)^{-1} X^T y
# Use np.linalg.inv as requested; in practice np.linalg.pinv is more stable.
theta_closed = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
intercept_closed, slope_closed = theta_closed[0], theta_closed[1]

print("Closed-form (Normal Equation):")
print(f"  Intercept = {intercept_closed:.4f}")
print(f"  Slope     = {slope_closed:.4f}")

# ---------------------------
# 3. Gradient Descent
# ---------------------------
theta = np.zeros(2)          # initialize theta = [0, 0]
learning_rate = 0.05         # eta
iterations = 1000
losses = []                  # store MSE loss each iteration

for i in range(iterations):
    preds = X.dot(theta)            # predictions
    error = preds - y               # residuals
    grad = (2 / m) * X.T.dot(error) # gradient of MSE
    theta = theta - learning_rate * grad

    mse = np.mean(error ** 2)
    losses.append(mse)

intercept_gd, slope_gd = theta[0], theta[1]
print("\nGradient Descent (after 1000 iters):")
print(f"  Intercept = {intercept_gd:.4f}")
print(f"  Slope     = {slope_gd:.4f}")

# ---------------------------
# 4. Plots: fitted lines + loss curve
# ---------------------------
# For clean lines, evaluate on sorted x values
x_line = np.linspace(0, 5, 200)
X_line = np.column_stack((np.ones_like(x_line), x_line))
y_line_closed = X_line.dot(theta_closed)
y_line_gd = X_line.dot(theta)

plt.figure(figsize=(7,5))
plt.scatter(x, y, alpha=0.4, label='Raw data')
plt.plot(x_line, y_line_closed, color='red', label='Closed-form fit')
plt.plot(x_line, y_line_gd, color='green', linestyle='--', label='Gradient Descent fit')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison: Closed-form vs Gradient Descent")
plt.legend()
plt.tight_layout()
plt.savefig("comparison_fits.png")
# plt.show()

# Loss curve
plt.figure(figsize=(6,4))
plt.plot(range(1, iterations+1), losses)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent: Loss vs Iterations")
plt.tight_layout()
plt.savefig("loss_curve.png")
# plt.show()

# Save final numbers to a small text file (optional)
with open("results.txt", "w") as f:
    f.write("Closed-form theta: {:.6f}, {:.6f}\n".format(intercept_closed, slope_closed))
    f.write("GD final theta:     {:.6f}, {:.6f}\n".format(intercept_gd, slope_gd))

print("\nSaved plots: raw_data.png, comparison_fits.png, loss_curve.png")
print("Saved numeric results: results.txt")