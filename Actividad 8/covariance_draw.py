import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Mean and Covariance matrix
mean = np.array([2, 3])
cov = np.array([[1.0, 0.01],
                [0.01, 1.5]])

# Get eigenvalues and eigenvectors
eigenvals, eigenvecs = np.linalg.eigh(cov)

# Get angle for ellipse
angle = np.degrees(np.arctan2(*eigenvecs[:, 1][::-1]))

# Width and height of the ellipse (2 standard deviations)
width, height = 2 * 2 * np.sqrt(eigenvals)

# Draw
fig, ax = plt.subplots()
ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  edgecolor='red', fc='None', lw=2)
ax.add_patch(ellipse)

# Plot mean
ax.plot(*mean, 'ro')

# Set limits
ax.set_xlim(mean[0] - 5, mean[0] + 5)
ax.set_ylim(mean[1] - 5, mean[1] + 5)
ax.set_aspect('equal')
plt.grid()
plt.title("Covariance Ellipse")
plt.show()
