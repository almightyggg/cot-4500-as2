# __init__.py - Cubic Spline Interpolation Module

def compute_cubic_spline(x, y):
    """Computes cubic spline interpolation for given data points."""
    
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
    return A, b, x_vector

def print_results(A, b, x_vector):
    """Prints the results for Matrix A, Vector b, and Vector x."""
    print("Matrix A:")
    for row in A:
        print(row)

    print("\nVector b:")
    print(b)

    print("\nVector x:")
    print(x_vector)


    # Compute cubic spline interpolation
    A, b, x_vector = compute_cubic_spline(x, y)

    # Print results
    print_results(A, b, x_vector)
