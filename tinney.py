"""Module for performing Tinney matrix re-ordering.

This re-ordering is done before LU factorization to reduce the number
of fills needed for sparse factorization.
"""
import numpy as np
import pandas as pd
import networkx as nx
from tabulate import tabulate
import matplotlib.pyplot as plt


def tinney1(g):
    """Function to perform Tinney 1 re-ordering.

    In Tinney 1, nodes are simply sorted by degree.

    NOTE: This function does not track fills, as that was not req'd by
    the homework.

    :param g: networkx graph representing the system.
    """
    # Sort the graph by degree and put into a DataFrame.
    deg = pd.DataFrame(list(g.degree), columns=['Bus', 'Degree'])

    # Sort by degree then bus. That's it for Tinny 1.
    deg.sort_values(by=['Degree', 'Bus'], inplace=True)

    return deg


def tinney2(g):
    """Function to perform Tinney 2 re-ordering and track fills.

    In Tinney 2, after each node is chosen, we remove it and update the
    degree of its neighbors after performing fills.

    NOTE: Here, we'll be tracking fills.

    TODO: This currently doesn't guarantee that ties will be broken by
    bus number.
    """

    # Initialize list of fills and node ordering.
    fills = []
    order = []

    # Sort the graph by degree.
    deg = sorted(g.degree, key=lambda x: x[1], reverse=False)
    node, degree = deg[0]

    # Loop over nodes.
    while True:
        # Add this node to the ordering.
        order.append(node)

        # If we're down to one node, we're done.
        if len(g.nodes) == 1:
            # Exit the while loop.
            break

        # Grab neighbors for this node.
        neighbors = set(g.neighbors(node))

        # Remove the node from the graph.
        g.remove_node(node)

        # Loop over the removed node's neighbors.
        n = neighbors.pop()
        while True:
            # Get the set difference between this n's neighbors and all
            # the remaining neighbors for 'node'.
            set_diff = neighbors.difference(set(g.neighbors(n)))

            # Loop over the set difference and add fills + edges.
            for k in set_diff:
                # Edge is from node n to k.
                edge = (n, k)

                # Add edge to graph.
                g.add_edge(*edge)

                # Track fills.
                fills.append(edge)

            # Get the next neighbor.
            try:
                n = neighbors.pop()
            except KeyError:
                # Set is empty, leave the inner while loop.
                break

        # Sort the graph by degree. NOTE: It would be more efficient to
        # update the sorted degree listing as we go.
        deg = sorted(g.degree, key=lambda x: x[1], reverse=False)
        node, degree = deg[0]

    return order, fills


def get_adj_mat(y_bus_df):
    """Helper to get adjacency matrix"""
    # Grab all the columns except 'Name'
    cols = y_bus_df.columns.drop('Name')

    # Get adjacency matrix. True where nodes are connected, False
    # otherwise.
    adj_mat = ~y_bus_df[cols].isnull().values

    return adj_mat


def get_graph(y_bus_df):
    """Given Y-bus DataFrame, get a networkx graph representation.

    """
    # Get adjacency matrix.
    adj_mat = get_adj_mat(y_bus_df)

    # Set diagonal elements to False so that the graph degree is correct.
    adj_mat[np.diag_indices(adj_mat.shape[0])] = False

    # Get graph.
    g = nx.convert_matrix.from_numpy_array(adj_mat, parallel_edges=False,
                                           create_using=nx.Graph)
    return g


def main():
    """Code for homework 4.

    NOTE: We'll be relying on the fact the y-bus matrices that are being
    read from file have empty entries for non-connected elements.
    """
    ####################################################################
    # LOAD DATA

    # Load the Y-bus matrix data.
    # ybus_5 = pd.read_csv('hw4/5_bus_y_bus.csv', index_col='Number')
    ybus_37 = pd.read_csv('hw4/37_bus_y_bus.csv', index_col='Number')

    ####################################################################
    # GET GRAPHS

    # Get graph.
    # g_5 = get_graph(ybus_5)
    g_37 = get_graph(ybus_37)

    ####################################################################
    # Tinney 1

    # Pass matrix values to tinney1.
    # deg_5 = tinney1(g_5)
    deg_37 = tinney1(g_37)

    # Grab bus names
    # bus_order_5 = list(ybus_5.index[deg_5['Bus'].values])
    # print(tabulate([[x] for x in bus_order_5], headers=['Tinney 1: 5 Bus']))

    # bus_order_37 = list(ybus_37.index[deg_37['Bus'].values])
    print(tabulate([[x] for x in deg_37['Bus'] + 1],
                   headers=['Tinney 1: 37 Bus'], tablefmt="latex"))

    ####################################################################
    # Tinney 2

    # order_5, fills_5 = tinney2(g_5)
    order_37, fills_37 = tinney2(g_37)

    print('')
    print(tabulate([[x] for x in np.array(order_37) + 1],
                   headers=['Tinney 2: 37 Bus'], tablefmt="latex"))

    fills = np.array(fills_37) + 1
    print('\nFills for 37 bus:')
    print(tabulate(fills, headers=['From', 'To'], tablefmt="latex"))

    # Illustrate fills.
    adj_mat = get_adj_mat(ybus_37)

    fig, ax = plt.subplots(1, 1)
    x, y = np.where(adj_mat)
    plt.plot(x + 1, y + 1, linestyle='None', marker='.', color='b')
    ax.invert_yaxis()
    plt.plot(fills[:, 0], fills[:, 1], linestyle='None', marker='x', color='r')
    plt.plot(fills[:, 1], fills[:, 0], linestyle='None', marker='x', color='r')
    # Save figure.
    plt.savefig('tinney2.eps', type='eps')


if __name__ == '__main__':
    main()

