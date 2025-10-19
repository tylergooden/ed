import numpy as np

x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 5)
X, Y = np.meshgrid(x, y)
xycoords = np.dstack([X, Y])
print("xycoords", xycoords)

