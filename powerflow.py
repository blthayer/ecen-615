"""Module for solving power flow for ECEN 615 HW2, Fall 2018.

Package prerequisites (simply install with pip): numpy, pandas, tabulate

It is assumed that this script will be run from the directory in which
it resides, and that the files with names described by the constants
BUS_FILE, LINES_FILE, and XFMRS_FILE also live in the same directory.

For assignment details, see hw2_prompt.txt.
"""
# Standard library.
import sys
from os.path import join as opj

# Installed packages.
import numpy as np
import pandas as pd
from tabulate import tabulate

# Local code.
import lu

# Directory containing system data.
IN_DIR = 'powerflow_files'

# Files containing system data.
BUS_FILE = opj(IN_DIR, '5_bus_buses.csv')
LINES_FILE = opj(IN_DIR, '5_bus_lines.csv')
XFMRS_FILE = opj(IN_DIR, '5_bus_xfmrs.csv')

# File for printing output:
OUT_DIR = 'hw3'
OUT_FILE = opj(OUT_DIR, 'hw3_output.txt')

# Pipe output to file.
#sys.stdout = open(OUT_FILE, 'w')

# Constants.
MVA_BASE = 100
TOL = 0.1 / MVA_BASE


def main(bus_file=BUS_FILE, lines_file=LINES_FILE, xfmrs_file=XFMRS_FILE,
         out_file=OUT_FILE, use_taps=False, solver='lu', tablefmt='latex',
         flat_flag=True):
    """Solve the power flow.

    bus_file should have the following columns: Bus, Type, V_pu, P_G,
        Q_G, P_L, Q_L, Q_max, Q_min
    """

    # Read bus data.
    bus_data = pd.read_csv(bus_file, index_col='Bus')

    # Fill NaN's with 0's.
    bus_data.fillna(0, inplace=True)

    # Define non-swing buses.
    non_swing = bus_data['Type'] != 'Swing'
    swing_v = bus_data[~non_swing]['V_pu'].values[0]
    # Get number of non_swing buses.
    N = np.count_nonzero(non_swing)
    # Get boolean of PV buses.
    non_pv = bus_data[non_swing]['Type'] != 'PV'
    pv_v = bus_data[non_swing][~non_pv]['V_pu']
    # Build it out into the boolean for eliminating Jacobian cols/rows.
    pv_bool = np.concatenate((np.ones_like(non_pv, dtype=np.bool_),
                             non_pv.values))

    # Identity matrix as boolean for Jacobian calculations.
    I = np.identity(bus_data.shape[0], dtype=np.bool_)
    I_not = ~I
    # Get floating point representation of I_not.
    I_not_f = I_not.astype(float)

    # Get views into I and I_not that don't include the swing. p for prime.
    # NOTE: THIS HARD-CODES SWING INTO THE 0th INDEX.
    I_p = I[1:, 1:]
    I_not_p = I_not[1:, 1:]

    # Initialize flat start.
    v, theta = flat_start(bus_data, flat_flag)

    # Get the y_bus matrix.
    lines_data = pd.read_csv(lines_file)
    xfmrs_data = pd.read_csv(xfmrs_file)
    y_bus = get_y_bus(bus_data=bus_data, lines_data=lines_data,
                      xfmrs_data=xfmrs_data, use_taps=use_taps)
    # Grab polar components (using GSO book). For some reason, the
    # np.abs call returns a DataFrame, while the np.angle call returns
    # a np.ndarray
    y_mag = np.abs(y_bus).values
    y_angle = np.angle(y_bus)

    # Initialize full Jacobian (no swing, but including PV buses)
    J = np.zeros((2*N, 2*N))

    it_count = 0
    # Silly initialization so we start the loop.
    f_x = 1 + TOL

    # Open output file.
    f_out = open(out_file, 'w')

    print('Voltages and angles at each iteration:', file=f_out)

    while (not np.all(np.abs(f_x) < TOL)) and (it_count < 100):

        # Print at start of each iteration.
        # print('*' * 79)
        print('Iteration {}:'.format(it_count), file=f_out)
        print(tabulate({'Bus': bus_data.index.values,
                        'Voltage (pu)': v, 'Angle (degrees)': theta*180/np.pi},
                       headers='keys', tablefmt=tablefmt), file=f_out)
        print('', file=f_out)

        # Pre-compute Ykn * Vn.
        ykn_vn = v * y_mag
        # Pre-compute y_kn * v_k. Transposing to multiply column-wise.
        ykn_vk = (v * y_mag.T).T
        # Pre-compute our angle difference.
        theta_kn = get_theta_kn(theta)
        theta_diff = theta_kn - y_angle

        # Pre-compute sin/cos of our angle difference
        sin_mat = np.sin(theta_diff)
        cos_mat = np.cos(theta_diff)

        # Get power at each bus.
        p, q = get_bus_power(v, ykn_vn, cos_mat, sin_mat)

        # Gen - Load - bus power.
        p_m = bus_data['P_G'] - bus_data['P_L'] - p
        q_m = bus_data['Q_G'] - bus_data['Q_L'] - q

        # Stack the mismatches, sans swing bus.
        f_x = np.concatenate([p_m[non_swing], q_m[non_swing]])

        # Stack theta and v, sans swing bus.
        x = np.concatenate((theta[non_swing], v[non_swing]))

        # Compute off-diagonal of jacobian quadrants.
        # j1
        J[0:N, 0:N][I_not_p] = (ykn_vk * v * sin_mat)[1:, 1:][I_not_p]
        # j2
        J[0:N, N:][I_not_p] = (ykn_vk * cos_mat)[1:, 1:][I_not_p]
        # j3
        J[N:, 0:N][I_not_p] = (-ykn_vk * v * cos_mat)[1:, 1:][I_not_p]
        # j4
        J[N:, N:][I_not_p] = (ykn_vk * sin_mat)[1:, 1:][I_not_p]

        # Compute diagonal jacobian entries.
        # Sum across axis 1 to add rows.
        # j1
        J[0:N, 0:N][I_p] = (-v * (ykn_vn * I_not_f * sin_mat).sum(axis=1))[1:]
        # j2
        J[0:N, N:][I_p] = (v * y_mag[I] * np.cos(y_angle[I])
                           + (ykn_vn * cos_mat).sum(axis=1))[1:]
        # j3
        J[N:, 0:N][I_p] = (v * (ykn_vn * I_not_f * cos_mat).sum(axis=1))[1:]
        # j4.
        J[N:, N:][I_p] = (-v * y_mag[I] * np.sin(y_angle[I])
                          + (ykn_vn * sin_mat).sum(axis=1))[1:]

        # Remove the PV column + row from Jacobian, x, and f_x
        J_p = J[:, pv_bool]
        J_p = J_p[pv_bool, :]
        x = x[pv_bool]
        f_x = f_x[pv_bool]

        # Compute next iteration.
        if solver == 'numpy':
            x = x + np.linalg.solve(J_p, f_x)
        elif solver == 'lu':
            x = x + lu.solve(J_p, f_x)

        # Extract theta from x, add swing angle (0) back in.
        theta = np.array([0, *x[0:N]])
        # Extract v from x, add swing voltage and PV bus voltage back in.
        # This is terribly inefficient....
        v = np.zeros(N)
        v[non_pv] = x[N:]
        v[~non_pv] = pv_v
        v = np.array([swing_v, *v])

        # Update iteration counter.
        it_count += 1

    # Generator outputs:
    buses = [bus_data.index[0], *bus_data.iloc[1:].loc[~non_pv].index.values]
    gen_p = (np.array([-p_m.iloc[0],
                       *bus_data[non_swing][~non_pv]['P_G'].values])
             * MVA_BASE)
    gen_q = np.array([-q_m.iloc[0], *-q_m.iloc[1:].loc[~non_pv]]) * MVA_BASE
    print('Final Generator Outputs:', file=f_out)
    print(tabulate({'Generator Bus': buses,
                    'Active Power (MW)': gen_p,
                    'Reactive Power (Mvar)': gen_q}, headers='keys',
                   tablefmt=tablefmt), file=f_out)

    return theta, v


def get_y_bus(bus_data, lines_data, xfmrs_data, use_taps):
    """

    :param bus_data: pandas DataFrame with bus data.
    :param lines_data: pandas DataFrame with line data.
    :param xfmrs_data: pandas DataFrame with transformer data.
    :param use_taps: boolean. If True, off-nominal taps will be
                     accounted for. Ignored if False.
    :return: y_bus: numpy array with shape [n_buses, n_buses],
             representing the system Y bus matrix.

    Expected file headers (same for both):
        lines_file: From, To, R_pu, X_pu, G_pu, B_pu, MVA_max
        xfmrs_data: From, To, R_pu, X_pu, G_pu, B_pu, MVA_max
    """
    # Get number of buses.
    n = bus_data.shape[0]
    # Initialize y_bus. NAIVE version: not dealing with sparsity.
    y_bus = pd.DataFrame(np.zeros((n, n), dtype=np.complex_),
                         index=bus_data.index, columns=bus_data.index)

    # Combine the two DataFrames.
    elements = pd.concat([lines_data, xfmrs_data], sort=False)

    # Loop over all passive elements and add to the Y-bus.
    for row in elements.itertuples():
        # Grab From and To as indices into the Y-bus.
        f = getattr(row, 'From')
        t = getattr(row, 'To')

        # Extract r, x, g, and b for easy access.
        r = getattr(row, 'R_pu')
        x = getattr(row, 'X_pu')
        g = getattr(row, 'G_pu')
        b = getattr(row, 'B_pu')

        # Compute admittance between buses.
        y_series = 1 / (r + 1j*x)

        # Compute shunt admittance: [from, to].
        y_shunt = g + 1j*b

        # Extract the tap_ratio. Will be NaN/null if there's no
        # off-nominal taps ratio (like for a line).
        tap_ratio = getattr(row, 'Tap_ratio')

        # If we're considering off-nominal tap ratios, and this element
        # has an off-nominal ratio, alter admittance elements.
        if use_taps and (not pd.isnull(tap_ratio)):
            # ASSUMPTION: xfmrs_data is given such that the tap is on
            # the FROM bus.

            # TODO: Do shunts need to be scaled as in the pi-model
            # depiction?

            # Add admittance to the [from, from] diagonal element.
            y_bus.loc[f, f] += (y_series / tap_ratio**2) + (y_shunt / 2)

            # The [to, to] diagonal element is just like normal.
            y_bus.loc[t, t] += y_series + y_shunt / 2

            # Off-diagonals are scaled by tap_ratio.
            y_bus.loc[f, t] -= y_series / tap_ratio
            y_bus.loc[t, f] -= y_series / tap_ratio

        else:
            # Standard case, no off-nominal taps.

            # Add admittance to diagonal elements:
            for i in [f, t]:
                y_bus.loc[i, i] += y_series + y_shunt / 2

            # Subtract from off-diagonals.
            y_bus.loc[f, t] -= y_series
            y_bus.loc[t, f] -= y_series

    # Done, return.
    return y_bus


def get_theta_kn(theta):
    """Helper for getting theta_kn matrix"""
    # Get bus differences, k - n, where our dimensions are [k, n]
    return (theta * np.ones((theta.shape[0], theta.shape[0]))).T - theta


def get_bus_power(v, ykn_vn, cos_mat, sin_mat):

    # Axis = 1 to sum across columns. Transposing to multiply column
    # wise instead of row wise.
    p = (v * (ykn_vn * cos_mat).sum(axis=1).T).T
    q = (v * (ykn_vn * sin_mat).sum(axis=1).T).T

    return p, q


def flat_start(bus_data, flat_flag=True):
    """Using the bus data, formulate a flat start.

    Start at low voltage (0.25) if the flag is False.
    """

    # Initialize voltages to 1.
    v = np.ones(bus_data.shape[0])

    if not flat_flag:
        # Attempt to find low voltage solution by starting at 0.25 pu.
        v = v * 0.25

    # Replace voltages at PV buses.
    pv_buses = bus_data['Type'] == 'PV'
    v[pv_buses] = bus_data[pv_buses]['V_pu'].values

    # Ensure the slack bus initializes correctly
    swing_bus = bus_data['Type'] == 'Swing'
    v[swing_bus] = bus_data[swing_bus]['V_pu'].values

    # Initialize angles to 0. Calculated everywhere except the swing.
    theta = np.zeros(bus_data.shape[0])

    return v, theta


if __name__ == '__main__':
    # Homework 3:
    # Problem 1: Use off-nominal taps and the numpy solver.
    main(use_taps=True, out_file=opj(OUT_DIR, 'hw3_problem1_output.txt'),
         solver='numpy')

    # Problem 3: Do not use off-nominal taps, and use my lu solver.
    main(use_taps=False, out_file=opj(OUT_DIR, 'hw3_problem3_output.txt'),
         solver='lu')
