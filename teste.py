import numpy as np
from numpy.linalg import lstsq

def find_y(xx):
    x = np.array([483.0, 376.5, 269.8, 162.9, 56.0])
    y = np.array([1.42,1.44,1.46,1.48,1.50])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = lstsq(A, y)[0]
    return m*xx+c


y=np.vectorize(find_y)

print(y([115.4,163.2,327.7,467.2,568.2]))
print(y([45.0,120.5,281.2,406.5,525.2]))
print(y([29.5,104.0,258.2,396.5,508.5]))

