"""Module for solving power flow for ECEN 615 HW2, Fall 2018.

For assignment details, see hw2_prompt.txt
"""
# Installed packages.
import numpy as np
import pandas as pd

# Files containing system data.
BUS_FILE = 'hw2_buses.csv'
LINES_FILE = 'hw2_lines.csv'
XFMRS_FILE = 'hw2_xfmrs.csv'

# Constants.
S_BASE = 100
TOL = 0.1/S_BASE

def main(bus_file=BUS_FILE, lines_file=LINES_FILE,
         xfmrs_file=XFMRS_FILE):
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
    v, theta = flat_start(bus_data)

    # Get the y_bus matrix.
    y_bus = get_y_bus(bus_data=bus_data, lines_file=lines_file,
                      xfmrs_file=xfmrs_file)
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

    while (not np.all(np.abs(f_x) < TOL)) and (it_count < 100):
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

        # Correct q for PV buses: should be 0.
        pv_q = f_x[~pv_bool]
        f_x[~pv_bool] = 0

        print('Mismatches at iteration {}:'.format(it_count))
        print(f_x)

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
        x = x + np.linalg.solve(J_p, f_x)

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


def get_y_bus(bus_data, lines_file, xfmrs_file):
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
    # Get number of buses.
    n = bus_data.shape[0]
    # Initialize y_bus. NAIVE version: not dealing with sparsity.
    y_bus = pd.DataFrame(np.zeros((n, n), dtype=np.complex_),
                         index=bus_data.index, columns=bus_data.index)

    # Read files.
    lines = pd.read_csv(lines_file)
    xfmrs = pd.read_csv(xfmrs_file)

    # Combine the two DataFrames.
    elements = pd.concat([lines, xfmrs])

    # Loop over all passive elements and add to the Y-bus.
    for _, row in elements.iterrows():
        # Grab From and To as indices into the Y-bus.
        f = row['From'].astype(int)
        t = row['To'].astype(int)

        # Compute admittance between buses.
        y = 1 / (row['R_pu'] + 1j*row['X_pu'])

        # Compute shunt admittance
        y_shunt = row['G_pu'] + 1j*row['B_pu']

        # Add admittance to diagonal elements:
        for i in [f, t]:
            y_bus.loc[i, i] += (y + y_shunt/2)

        # Subtract from off-diagonals.
        y_bus.loc[f, t] -= y
        y_bus.loc[t, f] -= y

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


def flat_start(bus_data):
    """Using the bus data, formulate a flat start."""

    # Initialize voltages to 1.
    v = np.ones(bus_data.shape[0])

    # Replace voltages at PV buses.
    pv_buses = bus_data['Type'] == 'PV'
    v[pv_buses] = bus_data[pv_buses]['V_pu'].values

    # Initialize angles to 0. Calculated everywhere except the swing.
    theta = np.zeros(bus_data.shape[0])

    return v, theta


if __name__ == '__main__':
    main()
