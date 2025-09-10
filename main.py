import numpy as np
import matplotlib.pyplot as plt
# first: Generate Synthetic 
# We are generating data based on: y = 3 + 4x + noise 
np.random.seed(0)  
# Generate 200 data points 
x = np.linspace(0, 5, 200)
noise = np.random.normal(0, 1, size=x.shape)  # Gaussian noise
y = 3 + 4 * x + noise  
# Plot
plt.scatter(x, y, label='Raw Data', color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Generated Data")
plt.legend()
plt.savefig("raw_data.png")
plt.close()

# 2nd: Closed-Form Solution 
X = np.c_[np.ones(x.shape[0]), x]  # Shape will be (200, 2)

# Normal Equation formula:θ = (X^T * X)^(-1) * X^T * y
theta_closed_form = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# The intercept and slope
intercept_cf, slope_cf = theta_closed_form

print("Closed-form (Normal Equation):")
print(f"  Intercept = {intercept_cf:.4f}")
print(f"  Slope     = {slope_cf:.4f}\n")

# Plot the fitted line using closed-form 
y_pred_cf = intercept_cf + slope_cf * x
plt.scatter(x, y, label='Raw Data', color='blue')
plt.plot(x, y_pred_cf, label='Closed-form Fit', color='green')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Closed-form Solution")
plt.legend()
plt.savefig("comparison_fits.png")
plt.close()

# 3rd: Gradient Descent Implementation

# Initialize 
theta_gd = np.zeros(2)  

# Learning rate 
learning_rate = 0.05
iterations = 1000

#(MSE)
loss_history = []

#Descent Loop
for i in range(iterations):
    
    y_pred = X.dot(theta_gd)

    error = y_pred - y

    gradient = (2 / len(y)) * X.T.dot(error)

    theta_gd -= learning_rate * gradient

    mse = np.mean(error ** 2)
    loss_history.append(mse)

intercept_gd, slope_gd = theta_gd

print("Gradient Descent (after 1000 iters):")
print(f"  Intercept = {intercept_gd:.4f}")
print(f"  Slope     = {slope_gd:.4f}\n")

# 4th: Plot Loss Curve

plt.plot(range(iterations), loss_history, color='red')
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Loss Curve for Gradient Descent")
plt.savefig("loss_curve.png")
plt.close()

# 5th: Save Final Results

with open("results.txt", "w") as f:
    f.write("Closed-form Solution:\n")
    f.write(f"Intercept: {intercept_cf:.4f}, Slope: {slope_cf:.4f}\n\n")
    f.write("Gradient Descent Solution:\n")
    f.write(f"Intercept: {intercept_gd:.4f}, Slope: {slope_gd:.4f}\n")

# 6th: Plot Both Fits for Comparison

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
