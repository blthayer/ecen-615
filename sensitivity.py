"""Module for ECEN 615 Homework 4.

NOTE: Problems 1 and 7 are in ../tinney.py.

NOTE: Throughout this module, assuming lines are numbered 1-N, and
therefore indices are 0-(N-1)
"""
import pandas as pd
import numpy as np

LINE_DATA_FILE = 'B7Flat_DC_lines.csv'


def get_b_tilde(data):
    """Compute the b_tilde matrix, which is defined as -diag{b1, b2, b_l}"""
    # Initialize to square matrix.
    b_tilde = np.zeros((data.shape[0], data.shape[0]))
    # Compute b values as simple 1/X (assuming DC). No need for -1 here,
    # taking a shortcut and not doing (1/(0+jX)).imag
    b_values = 1 / data['X'].values
    # Get indices of diagonal elements.
    i = np.diag_indices(data.shape[0])
    # Fill diagonal with b_values.
    b_tilde[i] = b_values
    # Return.
    return b_tilde


def get_a(data, slack_num=7):
    """Get the LxN incidence matrix (L is for each line, N is for nodes)

    NOTE: Sign convention is from lecture: positive for higher numbered,
    bus, negative for lower numbered bus.
    """
    # Initialize A. NOTE: This is not an efficient way to get the
    # number of nodes, but given our input data is only line values,
    # so be it.
    a = np.zeros((data.shape[0],
                  pd.unique(data[['From', 'To']].values.ravel('K')).shape[0]))

    # Loop over the rows in the data. Assume it's sorted by line.
    # Also assume line numberings start at 1.
    for row in data.itertuples():
        # Extract line (line), from (f), and to (t). Subtract by 1 to
        # convert numbers to indices.
        line = row.Line - 1
        f = row.From - 1
        t = row.To - 1
        # a[line, f] = 1
        # a[line, t] = -1
        if row.From < row.To:
            a[line, f] = 1
            a[line, t] = -1
        else:
            a[line, f] = -1
            a[line, t] = 1

    # Remove the slack bus column. This is not the efficient way to
    # handle this.
    a = np.hstack((a[:, 0:slack_num-1], a[:, slack_num:]))

    return a


def get_b_prime(a, b_tilde):
    """Helper to get the B' matrix"""
    return np.matmul(np.matmul(a.T, b_tilde), a)


def get_isf(data, slack_num):
    """Compute injection shift factor matrix."""
    # Compute the b_tilde matrix.
    b_tilde = get_b_tilde(data)

    # Compute incidence matrix a.
    a = get_a(data, slack_num=slack_num)

    # Compute B' matrix.
    b_prime = get_b_prime(a, b_tilde)

    # Compute the injection shift factor matrix. NOTE: doing the naive
    # thing and just taking the matrix inverse.
    return np.matmul(np.matmul(b_tilde, a), np.linalg.inv(b_prime))


def get_ptdf(isf, m, n, slack_num):
    """Power transfer distribution factors (PTDF) for a transaction.

    isf: injection shift factor matrix
    m: from bus, where power is injected
    n: to bus, where power is withdrawn
    slack_num: number of the slack bus, needed if the transaction
               involves the slack bus.
    """
    # Adjust numbers by one to get indices.
    m_ind = m - 1
    n_ind = n - 1

    # Moderately gross if/else to check for slack usage. Could be
    # factored further.
    if m == slack_num:
        psi_m = np.zeros((isf.shape[0], 1))
        psi_n = isf[:, n_ind]
    elif n == slack_num:
        psi_n = np.zeros(isf.shape[0])
        psi_m = isf[:, m_ind]
    else:
        psi_m = isf[:, m_ind]
        psi_n = isf[:, n_ind]

    # Compute and return ptdf.
    return psi_m - psi_n


def get_lodf(isf, m, n, slack_num, line_num):
    """Line outage distribution factors

    NOTE: m and n are the buses associated with line_num.
    """
    # Get power transfer distribution factors for the buses associated
    # with the line.
    ptdf = get_ptdf(isf=isf, m=m, n=n, slack_num=slack_num)

    # Compute and return the lodfs
    lodf = ptdf / (1 - ptdf[line_num - 1])

    # The position associated with this line should be -1.
    lodf[line_num - 1] = -1

    return lodf


def latex_b_matrix(a):
    """ Source: https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    return '\n'.join(rv)


def main(slack_num=7):
    # Read the line data. NOTE: It's already sorted by line.
    data = pd.read_csv(LINE_DATA_FILE)

    ####################################################################
    # COMPUTE INJECTION SHIFT FACTOR MATRIX (problem 3)
    isf = get_isf(data, slack_num)

    # Print the isf for Latex:
    print('Injection Shift Factor Matrix:')
    print(latex_b_matrix(np.round(isf, decimals=4)))
    print('\n')

    ####################################################################
    # PTDFs FOR ALL LINES FOR TRANSFER FROM BUS 2 TO BUS 7 (SLACK)
    # Since 7 is our slack, we simply pull the bus 2 column of the isf.
    # For another bus, it would be isf[:, bus_m] - isf[:, bus_n]
    # PTDF for power transfer distribution factor.
    ptdf = get_ptdf(isf, m=2, n=7, slack_num=slack_num)

    print('Power Transfer Distribution Factors, Transaction from Bus 2 to '
          'Bus 7:')
    print(latex_b_matrix(np.round(np.reshape(ptdf, (ptdf.shape[0], 1)),
                                  decimals=4)))
    print('\n')

    ####################################################################
    # LODFs FOR ALL LINES, OUTAGE BETWEEN BUS 2 AND 5
    # Get PTDF for line from 2 to 5.
    line_2_5 = 5
    # Hard-code the line number from 2 to 5
    lodf_2_5 = get_lodf(isf=isf, m=2, n=5, slack_num=slack_num,
                        line_num=line_2_5)

    # Reshape it.
    lodf1 = np.reshape(lodf_2_5, (lodf_2_5.shape[0], 1))

    print('Line Outage Distribution Factors for Outage Between Bus 2 and '
          'Bus 5:')
    print(latex_b_matrix(np.round(lodf1, decimals=4)))
    print('\n')

    ####################################################################
    # LODFs FOR ALL LINES, DOUBLE OUTAGE: 2-5 and 2-4
    # Gather bus definitions.
    m2 = 2
    n2 = 4
    line_2_4 = 4
    # We already have lodf_2_5 from the previous section. Get 2-4.
    lodf_2_4 = get_lodf(isf=isf, m=m2, n=n2, slack_num=slack_num,
                        line_num=line_2_4)

    # Reshape it.
    lodf2 = np.reshape(lodf_2_4, (lodf_2_4.shape[0], 1))

    # Build the 2x2 matrix.
    d = np.array([[1, -lodf2[line_2_5 - 1, 0]],
                  [-lodf1[line_2_4 - 1, 0], 1]])

    # Compute the lodf's for the double-outage.
    lodf_double = np.matmul(np.hstack((lodf1, lodf2)), np.linalg.inv(d))

    print('LODF For Double Outage: 2-5 and 2-4:')
    print(latex_b_matrix(np.round(lodf_double, decimals=4)))


if __name__ == '__main__':
    # The slack bus for this case is bus 7.
    main(slack_num=7)
