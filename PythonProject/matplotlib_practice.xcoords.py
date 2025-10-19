import matplotlib as plt
import numpy as np

#example dots on a line
xcoords = np.linspace(0, 1, 5)
plt.plot(xcoords, [0]*5, 'red')
plt.title("5 divisions; 01 to 1")
plt.show()

#example as grid
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 5)
X, Y = np.meshgrid(x, y)

plt.plot(X, Y, 'blue')
plt.title("5x5 grid of coordinate")
plt.show()

