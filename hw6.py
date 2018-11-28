"""Module for HW6"""
import numpy as np
import os
import powerflow

# Directories
DIR = 'hw6'
BUS_FILE = os.path.join(DIR, 'buses.csv')
BUS_FILE_2 = os.path.join(DIR, 'buses2.csv')
LINE_FILE = os.path.join(DIR, 'lines.csv')
XFMR_FILE = os.path.join(DIR, 'xfmrs.csv')
OUT_FILE = os.path.join(DIR, 'powerflow_output.txt')
OUT_FILE_HIGH = os.path.join(DIR, 'powerflow_output_high.txt')
OUT_FILE_LOW = os.path.join(DIR, 'powerflow_output_low.txt')

# Strings for creating files.
BUS_STR = ('Bus,Type,V_pu,P_G,Q_G,P_L,Q_L,Q_max,Q_min\n'
           + '1,Swing,1,,,,,,\n'
           + '2,PQ,,0,0,{},{},,')

LINE_STR = ('From,To,R_pu,X_pu,G_pu,B_pu,MVA_max\n'
            + '1,2,0,0.1,0,0,0')
XFMR_STR = 'From,To,R_pu,X_pu,G_pu,B_pu,MVA_max,Tap_ratio'


def problem1():
    # Power
    p = 4.54
    # p = 4.5
    # Admittance
    b = -10

    # Solve for q
    # -(p^2)/b + q + b/4 = 0
    q = p**2/b - b/4

    print('Value of Q_L: {:.4f}'.format(q))

    # Write to file.
    with open(BUS_FILE, 'w') as f:
        f.write(BUS_STR.format(p, q))

    # Solve the powerflow.
    t, v = powerflow.main(bus_file=BUS_FILE, lines_file=LINE_FILE,
                          xfmrs_file=XFMR_FILE, out_file=OUT_FILE,
                          use_taps=False, solver='numpy', tablefmt='simple')

    # Just grab theta and v for bus 2.
    t = t[1]
    v = v[1]

    print('Power flow solved with P_L = {:.4f} and Q_L = {:.4f}:'.format(p, q))
    print('\tTheta: {:.4f} radians, V: {:.4f} pu'.format(t, v))

    # Construct Jacobian (more hard-coding).
    jac = np.array([[-b*v*np.cos(t), -b*np.sin(t)],
                    [-b*v*np.sin(t), b*np.cos(t) - 2*b*v]])

    # Get left eigenvalues and eigenvectors (hence transposing the
    # Jacobian).
    values, vectors = np.linalg.eig(jac.T)

    # The normal is the left-hand eigenvector corresponding to the 0
    # eigenvalue. Due to rounding error, we aren't exactly getting 0.
    normal = vectors[:, np.argmin(values)]
    print('Normal vector to max loadability surface: {}'.format(normal))

    scalar = 0.25
    print('Scale the normal vector by {}, add to [P_L; Q_L].'.format(scalar))
    # Adjust load by normal.
    p_new = p + scalar * normal[0]
    q_new = q + scalar * normal[1]

    print('New P_L: {:.4f}. New Q_L: {:.4f}.'.format(p_new, q_new))

    # Write to file
    with open(BUS_FILE_2, 'w') as f:
        f.write(BUS_STR.format(p_new, q_new))

    print('Solve power flow twice: once to find "high" solution and once to '
          'find "low" solution.')

    # Find high voltage solution with new p/q.
    t_high, v_high = powerflow.main(bus_file=BUS_FILE_2, lines_file=LINE_FILE,
                                    xfmrs_file=XFMR_FILE,
                                    out_file=OUT_FILE_HIGH,
                                    use_taps=False, solver='numpy',
                                    tablefmt='simple')

    print('High voltage solution:')
    print('\tTheta: {:.4f} radians, V: {:.4f} pu'.format(t_high[1], v_high[1]))

    # Find low voltage solution with new p/q.
    t_low, v_low = powerflow.main(bus_file=BUS_FILE_2, lines_file=LINE_FILE,
                                  xfmrs_file=XFMR_FILE, out_file=OUT_FILE_LOW,
                                  use_taps=False, solver='numpy',
                                  tablefmt='simple',
                                  flat_flag=False)

    print('Low voltage solution:')
    print('\tTheta: {:.4f} radians, V: {:.4f} pu'.format(t_low[1], v_low[1]))
    pass


if __name__ == '__main__':
    # Write line and transformer files.

    # Start with the directory.
    try:
        os.mkdir(DIR)
    except FileExistsError:
        # No worries if the directory already exists.
        pass

    # Write files.
    with open(LINE_FILE, 'w') as f:
        f.write(LINE_STR)

    with open(XFMR_FILE, 'w') as f:
        f.write(XFMR_STR)

    # Run code for problem 1.
    problem1()
