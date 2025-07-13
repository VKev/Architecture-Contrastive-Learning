import numpy as np

A = np.random.rand(3, 3)
B = np.random.rand(3, 3)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

H, W = A.shape
X, Y = np.meshgrid(np.arange(W), np.arange(H))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Z is the base height (from A)
Z = A

# Difference vectors (0 in x and y, difference in z)
U = np.zeros_like(A)
V = np.zeros_like(A)
W = B - A  # Arrow from A to B

# Quiver: position (X,Y,Z), direction (U,V,W)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color='blue')

# Optional: plot A and B points
ax.scatter(X, Y, A, color='black', label='Matrix A')
ax.scatter(X, Y, B, color='red', label='Matrix B')

ax.set_xlabel("X (Column)")
ax.set_ylabel("Y (Row)")
ax.set_zlabel("Value")
ax.set_title("3D Quiver: A to B at Each (i,j)")
ax.invert_yaxis()
ax.legend()
plt.show()
