"""Module for LU factorization and forward/backward substitution.

NOTE: First draft does not handle permutations/pivoting.

Notation: We want to solve Ax = b
A = LU
LUx = b
Ux = y
Ly = b

Q = L + U - I
"""
import numpy as np


def lu_factorization(a):
    """Perform LU factorization using Crout's algorithm.

    Reference: Computational Methods for Electric Power Systems 2nd ed.,
    by Mariesa L. Crow.

    Assumption: a is square.

    :param a: numpy ndarray, where a.shape[0] = a.shape[1]
    :return: q: numpy ndarray, same shape as a, containing L (lower
             triangular matrix) and U (upper triangular, unity on
             diagonal)
    """
    # Initialize q. (which will hold L and U)
    q = np.zeros_like(a, dtype=np.float64)

    # Grab size of matrix (assume square) for easy access.
    n = a.shape[0]

    # Loop over columns of a.
    for j in range(n):
        # Complete the jth column of q (jth column of L)
        for k in range(j, n):
            q[k, j] = a[k, j] - (q[k, 0:j] * q[0:j, j]).sum()

        # Exit if early on last iteration
        if j == (n-1):
            break

        # Complete jth row of Q (jth row of U)
        for k in range(j+1, n):
            q[j, k] = ((1 / q[j, j])
                       * (a[j, k] - (q[j, 0:j] * q[0:j, k]).sum()))

    # That's it.
    return q


def forward_sub(q, b):
    """Solve for dummy vector y in Ly = b via forward substitution.

    :param q: Defined as L + U - I. Should come from calling
              lu_factorization()
    :param b: Vector of known values in Ly = b. Should already be
              multiplied by permutation matrix.
    :return y: Vector y from Ly = b, as a numpy array.
    """
    y = np.zeros_like(b, dtype=np.float64)

    # Loop forward.
    for k in range(b.shape[0]):
        y[k] = (1 / q[k, k]) * (b[k] - (q[k, 0:k] * y[0:k]).sum())

    return y


def backward_sub(q, y):
    """Solve for x in Ux = y via backward substitution.

    :param q: Defined as L + U - I. Should come from calling
              lu_factorization()
    :param y: Dummy vector, solved from Ly = b by calling forward_sub().
    :return x: Solution to Ux = y, as a numpy array.
    """
    # Get size of vector.
    n = y.shape[0]

    # Initialize x.
    x = np.zeros(n, dtype=np.float64)

    # Loop backward.
    for k in range(y.shape[0]-1, -1, -1):
        x[k] = y[k] - (q[k, k+1:n] * x[k+1:n]).sum()

    return x


def solve(a, b):
    """Perform LU factorization, then solve via forward and backward
    substitution."""
    # Perform LU decomposition.
    q = lu_factorization(a)

    # Perform forward substitution to solve for dummy vector y.
    y = forward_sub(q, b)

    # Perform backward substitution to solve for x.
    x = backward_sub(q, y)

    return x


def hw_3_problem_2():
    # Solve the system Ax = b, given A and b.
    a = np.array([[5, 1, 0, -4],
                  [1, 4, 0, -3],
                  [0, 0, 3, -2],
                  [-4, -3, -2, 10]])

    b = np.array([1, 2, 3, 4])

    # Perform LU decomposition.
    q = lu_factorization(a)

    # Perform forward substitution to solve for dummy vector y.
    y = forward_sub(q, b)

    # Perform backward substitution to solve for x.
    x = backward_sub(q, y)

    print('x = {}'.format(x))

    # Confirm with numpy.
    x_expected = np.linalg.solve(a, b)

    print('x_expected = {}'.format(x_expected))


def main():
    # Solve problem 2 on homework 3.
    hw_3_problem_2()

    # TODO: Examples from Crow could be used for unit testing.
    # # From Crow:
    # a = np.array([[1, 3, 4, 8],
    #               [2, 1, 2, 3],
    #               [4, 3, 5, 8],
    #               [9, 2, 7, 4]], dtype=np.float64)
    #
    # b = np.ones(a.shape[0], dtype=np.float64)
    #
    # # Get q.
    # q = lu_factorization(a)
    #
    # # Expected:
    # # q_expected = np.array([[1, 3, 4, 8],
    # #                        [2, -5, 1.2, 2.6],
    # #                        [4, -9, -0.2, 3],
    # #                        [9, -25, 1, -6]])
    #
    # # Forward sub for y.
    # y = forward_sub(q, b)
    # print('y: {}'.format(y))
    #
    # # Expected:
    # # y_expected = np.array([1, 0.2, 6, 1.5])
    #
    # # Backward sub for x.
    # x = backward_sub(q, y)
    # print('x: {}'.format(x))
    #
    # # Expected:
    # # x_expected = np.array([-0.5, -5.5, 1.5, 1.5])


if __name__ == '__main__':
    main()
