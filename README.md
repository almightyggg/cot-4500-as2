# cot-4500-as2

This assignment will cover chapter 3. Heads up, this requires tons of programming. In Chapter 3,
we covered interpolating polynomials. Polynomial interpolation is the interpolation of a given
data set by the polynomial of lowest possible degree that passes through the points of the
dataset.
This helps us approximate points on complicated curves. The methods require tons of
operations on different points along the curve and introduces matrix operations.

python -m cot_4500_as2

pip install
scipy (for interpolation and advanced numerical methods)

from cot-4500-as2 import compute_cubic_spline

x = [2, 5, 8, 10]
y = [3, 5, 7, 9]

A, b, x_vector = compute_cubic_spline(x, y)

print("Matrix A:")
for row in A:
    print(row)

print("\nVector b:")
print(b)

print("\nVector x:")
print(x_vector)
