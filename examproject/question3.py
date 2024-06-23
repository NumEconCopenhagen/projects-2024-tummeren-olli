import numpy as np
import matplotlib.pyplot as plt

# We define the function f(x1, x2) = x1 * x2
def f(x):
    return x[0] * x[1]

# We set up a function that finds the points A, B, C, and D.
# We set it up so it takes the set of random points X and the point y as input. Then it uses the conditions to find A, B, C, and D.
def find_points(X, y):
    A = min([x for x in X if x[0] > y[0] and x[1] > y[1]], 
            key=lambda x: np.linalg.norm(x - y), default=None)
    B = min([x for x in X if x[0] > y[0] and x[1] < y[1]], 
            key=lambda x: np.linalg.norm(x - y), default=None)
    C = min([x for x in X if x[0] < y[0] and x[1] < y[1]], 
            key=lambda x: np.linalg.norm(x - y), default=None)
    D = min([x for x in X if x[0] < y[0] and x[1] > y[1]], 
            key=lambda x: np.linalg.norm(x - y), default=None)
    return A, B, C, D

# We then set up the function that plots the points and the triangles ABC and CDA.
def plot_points_and_triangles(X, y, A, B, C, D):
    plt.scatter(X[:, 0], X[:, 1], label='Points in X')
    plt.scatter(y[0], y[1], color='red', label='Point y')
    if A is not None and B is not None and C is not None:
        plt.scatter(A[0], A[1], color='blue', label='Point A')
        plt.scatter(B[0], B[1], color='green', label='Point B')
        plt.scatter(C[0], C[1], color='purple', label='Point C')
        plt.plot([A[0], B[0]], [A[1], B[1]], 'k-')
        plt.plot([B[0], C[0]], [B[1], C[1]], 'k-')
        plt.plot([C[0], A[0]], [C[1], A[1]], 'k-')
    if C is not None and D is not None and A is not None:
        plt.scatter(D[0], D[1], color='orange', label='Point D')
        plt.plot([C[0], D[0]], [C[1], D[1]], 'k--')
        plt.plot([D[0], A[0]], [D[1], A[1]], 'k--')
        plt.plot([A[0], C[0]], [A[1], C[1]], 'k--')
    plt.legend()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Points and Triangles')
    plt.grid(True)
    plt.show()

# We set up the function that computes the barycentric coordinates of y with respect to the triangle ABC.
def barycentric_coordinates(y, A, B, C):
    denominator = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
    r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denominator
    r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denominator
    r3 = 1 - r1 - r2
    return r1, r2, r3

# We then check if the point we found above is inside the triangle.
# We do this by checking if the barycentric coordinates are between 0 and 1.
def is_inside_triangle(r1, r2, r3):
    return 0 <= r1 <= 1 and 0 <= r2 <= 1 and 0 <= r3 <= 1

# We find the interpolation by computing the interpolated value of f at y with a barycentric interpolation.
# We set it up so if any of the points A, B, C, or D are None, it returns NaN.
def interpolate(y, A, B, C, D, f):
    if A is None or B is None or C is None or D is None:
        return np.nan
    r_ABC = barycentric_coordinates(y, A, B, C)
    if is_inside_triangle(*r_ABC):
        f_A = f(A)
        f_B = f(B)
        f_C = f(C)
        return r_ABC[0] * f_A + r_ABC[1] * f_B + r_ABC[2] * f_C
 # The above code is the same as the one below, but with the points A, B, and C switched with C, D, and A. 
 # So in the above, if y is inside the triangle ABC, we compute the interpolated value by using the barycentric coordinates and the f value at A, B, and C. 
    r_CDA = barycentric_coordinates(y, C, D, A)
    if is_inside_triangle(*r_CDA):
        f_C = f(C)
        f_D = f(D)
        f_A = f(A)
        return r_CDA[0] * f_C + r_CDA[1] * f_D + r_CDA[2] * f_A
    
    return np.nan