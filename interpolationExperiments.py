"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Define some points:
points = np.array([[0,0], [0.75, 0.25], [0.25, 0.75], [1.0, 1.0]])

# Linear length along the line:
distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
distance = np.insert(distance, 0, 0)/distance[-1]
print(distance)

# Interpolation for different methods:
interpolations_methods = ['slinear', 'cubic']
alpha = np.linspace(0, 1, 75)

interpolated_points = {}
for method in interpolations_methods:
    interpolator =  interp1d(distance, points, kind=method, axis=0)
    interpolated_points[method] = interpolator(alpha)

# Graph:
plt.figure(figsize=(7,7))
for method_name, curve in interpolated_points.items():
    plt.plot(*curve.T, 'o', label=method_name);

plt.plot(*points.T, 'ok', label='original points');
plt.axis('equal'); plt.legend(); plt.xlabel('x'); plt.ylabel('y');
plt.show()
""" 
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

# Define some points:
theta = np.linspace(-3, 2, 40)
points = np.array([[0.0, 0.0], [0.75, 0.25], [0.25, 0.75], [1.0, 1.0]])

# add some noise:
points = points + 0.05*np.random.randn(*points.shape)

# Linear length along the line:
distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
distance = np.insert(distance, 0, 0)/distance[-1]
# distance = [0, 0.5, 0.7, 1.0]

# Build a list of the spline function, one for each dimension:
splines = [UnivariateSpline(distance, coords, k=3, s=.2) for coords in points.T]

# Computed the spline for the asked distances:
alpha = np.linspace(0, 1, 75)
points_fitted = np.vstack( spl(alpha) for spl in splines ).T

# Graph:
plt.plot(*points.T, 'ok', label='original points');
plt.plot(*points_fitted.T, 'or', label='fitted spline k=3, s=.2');
plt.axis('equal'); plt.legend(); plt.xlabel('x'); plt.ylabel('y');
plt.show()