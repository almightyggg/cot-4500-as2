import numpy as np
import scipy.interpolate as interp

# 1. Neville's Method
def neville_interpolation(x_vals, y_vals, x):
    n = len(x_vals)
    Q = np.zeros((n, n))
    Q[:, 0] = y_vals

    for j in range(1, n):
        for i in range(n - j):
            Q[i, j] = ((x - x_vals[i + j]) * Q[i, j - 1] + (x_vals[i] - x) * Q[i + 1, j - 1]) / (x_vals[i] - x_vals[i + j])
    
    return Q[0, -1]

x_vals = [3.6, 3.8, 3.9]
y_vals = [1.675, 1.436, 1.318]
x_target = 3.7
neville_result = neville_interpolation(x_vals, y_vals, x_target)
print(f"Neville's Method Output: {neville_result:.15f}")

# 2. Newton's Forward Difference Method
def newton_forward_difference(x_vals, y_vals):
    n = len(x_vals)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_vals

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = (diff_table[i+1, j-1] - diff_table[i, j-1]) / (x_vals[i+j] - x_vals[i])

    return diff_table

x_vals_newton = [7.2, 7.4, 7.5, 7.6]
y_vals_newton = [23.5492, 25.3913, 26.8224, 27.4589]
newton_table = newton_forward_difference(x_vals_newton, y_vals_newton)

print("Newton's Forward Difference Table:")
print(" ".join(f"{val:.15f}" for val in newton_table[0] if val != 0))


# 3. Newton's Interpolation Approximation
def newton_interpolation(x_vals, y_vals, x_target):
    n = len(x_vals)
    divided_diff = newton_forward_difference(x_vals, y_vals)[0, :]

    approx = divided_diff[0]
    product_term = 1.0

    for i in range(1, n):
        product_term *= (x_target - x_vals[i-1])
        approx += divided_diff[i] * product_term
    
    return approx

newton_approx_7_3 = newton_interpolation(x_vals_newton, y_vals_newton, 7.3)
print(f"Newton's Approximation for f(7.3): {newton_approx_7_3:.15f}")

# 4. Hermite Polynomial Approximation
def hermite_interpolation_matrix(x_vals, y_vals, y_derivs):
    n = len(x_vals)
    z = np.zeros(2*n)
    Q = np.zeros((2*n, 2*n))

    for i in range(n):
        z[2*i] = x_vals[i]
        z[2*i+1] = x_vals[i]
        Q[2*i, 0] = x_vals[i]
        Q[2*i+1, 0] = x_vals[i]
        Q[2*i, 1] = y_vals[i]
        Q[2*i+1, 1] = y_vals[i]
        Q[2*i+1, 2] = y_derivs[i]
        if i != 0:
            Q[2*i, 2] = (Q[2*i, 1] - Q[2*i-1, 1]) / (z[2*i] - z[2*i-1])

    for j in range(3, 2*n):
        for i in range(2*n - j):
            Q[i, j] = (Q[i+1, j-1] - Q[i, j-1]) / (z[i+j-1] - z[i])

    return Q

x_vals_hermite = [3.6, 3.8, 3.9]
y_vals_hermite = [1.675, 1.436, 1.318]
y_derivs_hermite = [-1.195, -1.188, -1.182]
hermite_matrix = hermite_interpolation_matrix(x_vals_hermite, y_vals_hermite, y_derivs_hermite)

print("Hermite Interpolation Matrix:")
for row in hermite_matrix:
    formatted_row = [f"{val:+.8e}" if val != 0 else "0.00000000e+00" for val in row]
    print("[", " ".join(formatted_row), "]")

# 5. Cubic Spline Interpolation
# Given data points
x = [2, 5, 8, 10]
y = [3, 5, 7, 9]

# Compute step sizes
h = [x[i+1] - x[i] for i in range(len(x) - 1)]

# Construct matrix A
n = len(x)
A = [[0] * n for _ in range(n)]
A[0][0] = 1
A[-1][-1] = 1

for i in range(1, n - 1):
    A[i][i - 1] = h[i - 1]
    A[i][i] = 2 * (h[i - 1] + h[i])
    A[i][i + 1] = h[i]

# Construct vector b
b = [0] * n
for i in range(1, n - 1):
    b[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

# Solve for vector x (second derivatives at each x) using Gaussian elimination
def gauss_elimination(A, b):
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            if A[i][i] == 0:
                continue
            ratio = A[j][i] / A[i][i]
            for k in range(n):
                A[j][k] -= ratio * A[i][k]
            b[j] -= ratio * b[i]
    
    x = [0] * n
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]
    return x

x_vector = gauss_elimination(A, b)

# Display results
print("Matrix A:")
for row in A:
    print(row)

print("\nVector b:")
print(b)

print("\nVector x:")
print(x_vector)

