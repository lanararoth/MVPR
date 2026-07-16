import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# -----------------------------
# Generate Synthetic Gaussian Data
# -----------------------------
np.random.seed(42)

# Mean and covariance for Class 1
mean1 = [2, 2]
cov1 = [[1, 0.5],
        [0.5, 1]]

# Mean and covariance for Class 2
mean2 = [6, 6]
cov2 = [[1, -0.3],
        [-0.3, 1]]

# Generate samples
class1 = np.random.multivariate_normal(mean1, cov1, 100)
class2 = np.random.multivariate_normal(mean2, cov2, 100)

# -----------------------------
# Prior Probabilities
# -----------------------------
P1 = 0.5
P2 = 0.5

# -----------------------------
# Create Grid for Decision Boundary
# -----------------------------
x, y = np.meshgrid(np.linspace(-1, 9, 200),
                   np.linspace(-1, 9, 200))

grid = np.dstack((x, y))

# Compute likelihoods
pdf1 = multivariate_normal(mean1, cov1).pdf(grid)
pdf2 = multivariate_normal(mean2, cov2).pdf(grid)

# Posterior probabilities (Bayes Rule)
posterior1 = pdf1 * P1
posterior2 = pdf2 * P2

# Classification
decision = posterior1 > posterior2

# -----------------------------
# Plot Decision Boundary
# -----------------------------
plt.figure(figsize=(8,6))

# Decision regions
plt.contourf(x, y, decision, alpha=0.3, cmap='coolwarm')

# Decision boundary
plt.contour(x, y,
            posterior1 - posterior2,
            levels=[0],
            colors='black',
            linewidths=2)

# Plot training samples
plt.scatter(class1[:,0], class1[:,1],
            color='blue',
            label='Class 1')

plt.scatter(class2[:,0], class2[:,1],
            color='red',
            label='Class 2')

plt.title("Bayes Classifier Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

plt.show()

# -----------------------------
# Classify Test Points
# -----------------------------
test_points = np.array([
    [3,3],
    [5,5],
    [2,5],
    [7,6]
])

print("Classification of Test Points:\n")

for point in test_points:

    p1 = multivariate_normal(mean1, cov1).pdf(point) * P1
    p2 = multivariate_normal(mean2, cov2).pdf(point) * P2

    predicted = "Class 1" if p1 > p2 else "Class 2"

    print("Point:", point)
    print("P(Class1|x):", round(p1,6))
    print("P(Class2|x):", round(p2,6))
    print("Predicted:", predicted)
    print()
    