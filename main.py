import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: Generate Synthetic Data
# ----------------------------
# We are generating data based on the formula:
# y = 3 + 4x + noise (Gaussian random noise added)
np.random.seed(0)  # For reproducibility, so results are always the same

# Generate 200 data points where x ranges from 0 to 5
x = np.linspace(0, 5, 200)
noise = np.random.normal(0, 1, size=x.shape)  # Gaussian noise
y = 3 + 4 * x + noise  # Target values with noise added

# Plot and save the raw dataset
plt.scatter(x, y, label='Raw Data', color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Generated Data")
plt.legend()
plt.savefig("raw_data.png")
plt.close()

# ----------------------------
# Step 2: Closed-Form Solution (Normal Equation)
# ----------------------------
# Add a column of 1's to x to represent the bias term (intercept)
# This creates the design matrix X
X = np.c_[np.ones(x.shape[0]), x]  # Shape will be (200, 2)

# Normal Equation formula:
# θ = (X^T * X)^(-1) * X^T * y
theta_closed_form = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Extract the intercept and slope
intercept_cf, slope_cf = theta_closed_form

print("Closed-form (Normal Equation):")
print(f"  Intercept = {intercept_cf:.4f}")
print(f"  Slope     = {slope_cf:.4f}\n")

# Plot the fitted line using closed-form solution
y_pred_cf = intercept_cf + slope_cf * x
plt.scatter(x, y, label='Raw Data', color='blue')
plt.plot(x, y_pred_cf, label='Closed-form Fit', color='green')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Closed-form Solution")
plt.legend()
plt.savefig("comparison_fits.png")
plt.close()

# ----------------------------
# Step 3: Gradient Descent Implementation
# ----------------------------
# Initialize parameters (θ0 for intercept, θ1 for slope)
theta_gd = np.zeros(2)  # [intercept, slope]

# Learning rate (η) and number of iterations
learning_rate = 0.05
iterations = 1000

# To track Mean Squared Error (MSE) during training
loss_history = []

# Gradient Descent Loop
for i in range(iterations):
    # Predicted y values
    y_pred = X.dot(theta_gd)

    # Error between predicted and actual y
    error = y_pred - y

    # Compute gradient (partial derivatives of the cost function)
    gradient = (2 / len(y)) * X.T.dot(error)

    # Update theta values
    theta_gd -= learning_rate * gradient

    # Calculate and store the Mean Squared Error for this iteration
    mse = np.mean(error ** 2)
    loss_history.append(mse)

# Final parameters after Gradient Descent
intercept_gd, slope_gd = theta_gd

print("Gradient Descent (after 1000 iters):")
print(f"  Intercept = {intercept_gd:.4f}")
print(f"  Slope     = {slope_gd:.4f}\n")

# ----------------------------
# Step 4: Plot Loss Curve
# ----------------------------
plt.plot(range(iterations), loss_history, color='red')
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Loss Curve for Gradient Descent")
plt.savefig("loss_curve.png")
plt.close()

# ----------------------------
# Step 5: Save Final Results to a Text File
# ----------------------------
with open("results.txt", "w") as f:
    f.write("Closed-form Solution:\n")
    f.write(f"Intercept: {intercept_cf:.4f}, Slope: {slope_cf:.4f}\n\n")
    f.write("Gradient Descent Solution:\n")
    f.write(f"Intercept: {intercept_gd:.4f}, Slope: {slope_gd:.4f}\n")

# ----------------------------
# Step 6: Plot Both Fits for Comparison
# ----------------------------
plt.scatter(x, y, label='Raw Data', color='blue')
plt.plot(x, y_pred_cf, label='Closed-form Fit', color='green')
plt.plot(x, intercept_gd + slope_gd * x, label='Gradient Descent Fit', color='red', linestyle='--')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison: Closed-form vs Gradient Descent")
plt.legend()
plt.savefig("comparison_fits.png")
plt.close()

print("Saved plots: raw_data.png, comparison_fits.png, loss_curve.png")
print("Saved numeric results: results.txt")
