# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file and store the data in a NumPy array
data = np.genfromtxt('/users/navneet/desktop/exponential.csv', delimiter=',')

# Extract the x and y dimensions from the array
x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))

# Extract the z-values from the array
z = data[x, y]

# Create the 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z)

# Show the plot
plt.show()
