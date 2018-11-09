"""Module for hw5 computations"""
import numpy as np
from numpy import sin, cos

# Threshold for condition number.
COND_THRESH = 1e10


def problem1():
    """Book problem 9.1"""
    # measurements:
    z = np.array([0.6, 0.04, 0.405])

    # H matrix (relate measurements to states):
    h = np.array([[5, -5], [2.5, -2.5], [0, 4]])

    # R matrix (std. deviation of measurements):
    r = np.zeros((3, 3))
    r[np.diag_indices(3)] = np.array([0.02**2, 0.01**2, 0.002**2])

    # Part a:
    # Solve for x.
    x = linear_se(r, h, z)
    print('Problem 1')
    print('States:')
    print(x)

    # Part b:
    # z - Hx
    zhx = z - np.matmul(h, x)
    # J(x)
    j_x = np.matmul(np.matmul(zhx.T, np.linalg.inv(r)), zhx)
    print('J(x):')
    print(j_x)
    print('\n')

    # Lookup value of Chi Squared from table.


def problem2():
    """Book problem 9.3"""
    # part a:
    z_a = np.array([-0.705, 0.721, 0.212])

    h_a = np.array([[2, 0, -2],
                   [-2, 0, 2],
                   [4, -4, 0]])

    r_a = np.zeros((3, 3))
    r_a[np.diag_indices(3)] = np.array([0.01**2, 0.01**2, 0.02**2])

    print('Problem 2')

    # The network is not observable.
    try:
        x_a = linear_se(r_a, h_a, z_a)
    except UserWarning:
        print('The system for part a is not observable.')
    else:
        print(x_a)

    # part b:
    z_b = np.concatenate((z_a, np.array([0.92 - 0.721])))
    h_b = np.vstack((h_a, np.array([0, 0, 10])))
    r_b = np.zeros((4, 4))
    r_b[np.diag_indices(4)] = np.array([0.01 ** 2, 0.01 ** 2, 0.02 ** 2,
                                        0.015**2])

    # Looking at it, it's observable and thus invertible.
    x_b = linear_se(r_b, h_b, z_b)
    print('States for part b:')
    print(x_b)


def problem3():
    """Perform 2nd and 3rd iterations of AC SE from lecture 19"""
    # Measurements
    z = np.array([2.02,   # P12
                  1.5,    # Q12
                  -1.98,  # P21
                  -1,     # Q21
                  1.01,   # V1
                  0.87    # V2
                  ])

    # States (guess):
    x = np.array([1, 0, 1])

    r = np.zeros((z.shape[0], z.shape[0]))
    np.fill_diagonal(r, 0.0001)

    print('Problem 3:')
    print('x0: {}'.format(x))

    # Loop.
    iterations = 0
    while iterations < 4:
        # Extract v1, theta2, and v3 for readability
        v1 = x[0]
        t2 = x[1]
        v2 = x[2]

        # compute h.
        h = np.array([
            [v2*10*sin(-t2),         -v1*v2*cos(-t2),    v1*10*sin(-t2)],
            [20*v1 - v2*10*cos(-t2), -v1*v2*10*sin(-t2), -v1*10*cos(-t2)],
            [v2*10*sin(t2),          v1*v2*10*cos(t2),   v1*10*sin(t2)],
            [-v2*10*cos(t2),         v1*v2*10*sin(t2),   20*v2-v1*10*cos(t2)],
            [1,                      0,                  0],
            [0,                      0,                  1]
        ])

        # Compute z - f(x)
        delta = z - np.array([
            v1 * v2 * 10 * sin(-t2),
            v1**2 * 10 + v1 * v2 * (-10 * cos(-t2)),
            v1 * v2 * 10 * sin(t2),
            v2 ** 2 * 10 + v1 * v2 * (-10 * cos(t2)),
            v1,
            v2
        ])

        # Compute parts of delta x = [H^T * R^-1 * H]^-1 * H^T * R^-1 * [z - f]
        # H^T * R^-1
        htr = np.matmul(h.T, np.linalg.inv(r))
        # [H^T * R^-1 * H]^-1
        htrh = np.linalg.inv(np.matmul(htr, h))
        # H^T * R^-1 * (z - f(x))
        htrd = np.matmul(htr, delta)

        # Compute delta x
        dx = np.matmul(htrh, htrd)

        # Update x
        x = x + dx

        # Update iteration counter
        iterations += 1

        print('x{}: {}'.format(iterations, x))


def givens_s_c(a, b):
    """Helper to compute s and c for Givens Rotation"""
    if abs(b) > abs(a):
        t = -a/b
        s = 1/((1 + t**2)**0.5)
        c = s*t
    else:
        t = -b/a
        c = 1/((1 + t**2)**0.5)
        s = c*t

    return s, c


def problem4():
    """Use Givens Rotation to perform QR factorization.

    This is just a manual (nasty) hard-code, not an attempt at a general
    implementation.
    """
    A = np.array([[1, 2], [3, 4], [5, 6]])

    # Zero out A[3, 1] (one-based indexing)
    i = 3 - 1
    j = 1 - 1
    a = A[i-1, j]
    b = A[i, j]
    s, c = givens_s_c(a, b)
    g1 = np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])

    print('G1:')
    print(g1)

    g1a = np.matmul(g1.T, A)
    g1a[np.isclose(g1a, 0)] = 0
    print('G1^T * A:')
    print(g1a)

    # Zero out A[2, 1]
    i = 2 - 1
    j = 1 - 1
    a = g1a[i-1, j]
    b = g1a[i, j]
    s, c = givens_s_c(a, b)
    g2 = np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])

    print('G2:')
    print(g2)

    g2a = np.matmul(g2.T, g1a)
    g2a[np.isclose(g2a, 0)] = 0
    print('G2^T * G1^T * A:')
    print(g2a)

    # Zero out A[3, 2]
    i = 3 - 1
    j = 2 - 1
    a = g2a[i-1, j]
    b = g2a[i, j]
    s, c = givens_s_c(a, b)
    g3 = np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])

    print('G3:')
    print(g3)

    g3a = np.matmul(g3.T, g2a)
    g3a[np.isclose(g3a, 0)] = 0
    print('G^3T * G2^T * G1^T * A:')
    print(g3a)

    # We're done, but need all of our G's multiplied together.
    print('G1 * G2 * G3:')
    g123 = np.matmul(np.matmul(g1, g2), g3)
    print(g123)


def linear_se(r, h, z):
    """Helper function for linear state estimation.

    x = [H^T * R^-1 * H]^-1 * H^T * R^-1 * z,
    where ^T is transpose and ^-1 is inverse.

    r: diagonal matrix of standard deviation
    h: matrix relating measurements to states
    z: measurements
    """
    # Get inverse of r.
    r_inv = np.linalg.inv(r)

    # H_transposed * R inverse
    htr = np.matmul(h.T, r_inv)

    # htr * z
    htrz = np.matmul(htr, z)

    # htr * h
    htrh = np.matmul(htr, h)

    # Check condition number. This is not efficient, but numpy doesn't
    # throw errors if it can "solve" a singular matrix.
    c = np.linalg.cond(htrh)

    if c > COND_THRESH:
        raise UserWarning('H^T * R^-1 * H is singular (maybe)!')

    # Solve for x.
    x = np.matmul(np.linalg.inv(htrh), htrz)

    return x


def main():
    # problem1()
    # problem2()
    # problem3()
    problem4()


if __name__ == '__main__':
    main()
