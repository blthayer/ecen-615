"""Module for solving power flow for ECEN 615 HW2, Fall 2018.

For assignment details, see hw2_prompt.txt
"""
import numpy as np
import pandas as pd


def main():
    # For now, hard-code 5. Should get it from the buses file, though.
    y_bus = get_y_bus(5, lines_file='hw2_lines.csv',
                      xfmrs_file='hw2_xfmrs.csv')

    pass


def get_y_bus(n_buses, lines_file, xfmrs_file):
    """

    :param n_buses: integer, number of buses in the system.
    :param lines_file: string, file with line data.
    :param xfmrs_file: string, file with transformer data.
    :return: y_bus: numpy array with shape [n_buses, n_buses],
             representing the system Y bus matrix.

    Expected file headers (same for both):
        lines_file: From, To, R_pu, X_pu, G_pu, B_pu, MVA_max
        xfmrs_file: From, To, R_pu, X_pu, G_pu, B_pu, MVA_max
    """
    # Initialize y_bus. NAIVE version: not dealing with sparsity.
    y_bus = np.zeros((n_buses, n_buses), dtype=np.complex_)

    # Read files.
    lines = pd.read_csv(lines_file)
    xfmrs = pd.read_csv(xfmrs_file)

    # Combine the two DataFrames.
    elements = pd.concat([lines, xfmrs])

    # Loop over all passive elements and add to the Y-bus.
    for _, row in elements.iterrows():
        # Grab From and To as indices into the Y-bus.
        f = (row['From'] - 1).astype(int)
        t = (row['To'] - 1).astype(int)

        # Compute admittance between buses.
        Y = 1 / (row['R_pu'] + 1j*row['X_pu'])

        # Compute shunt admittance
        Y_shunt = row['G_pu'] + 1j*row['B_pu']

        # Add admittance to diagonal elements:
        for i in [f, t]:
            y_bus[i, i] += (Y + Y_shunt/2)

        # Subtract from off-diagonals.
        y_bus[f, t] -= Y
        y_bus[t, f] -= Y

    # Done, return.
    return y_bus


if __name__ == '__main__':
    main()
