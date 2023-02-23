import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import time

# --------------- EDIDTING nodes_df ----------------#


def create_nodes_df():
    """
    This function creates and returns an empty DataFrame represetning nodes of
    following form:
                    x_coordinate  y_coordinate   required_capacity  max_power   shs_price
        label
        a           0.0           0.0             3.0               3.0         200
        b           0.0           1.0             6.0               6.0         300
        c           1.0           1.0             10.0              200.0       1000
        d           2.0           0.0             2.0               2.0         50

    Output
    ------
        Empty nodes DataFrame.
    """

    return pd.DataFrame(
        {
            'label':
            pd.Series(
                [],
                dtype=str),
            'x_coordinate':
            pd.Series([],
                      dtype=np.dtype(float)),
            'y_coordinate':
            pd.Series([],
                      dtype=np.dtype(float)
                      ),
            'required_capacity':
            pd.Series([],
                      dtype=np.dtype(float)
                      ),
            'max_power':
            pd.Series([], dtype=np.dtype(float)
                      ),
            'shs_price':
            pd.Series([], dtype=np.dtype(float)
                      )
        }
    ).set_index('label')


def add_node(nodes_df,
             node_label,
             x_coordinate,
             y_coordinate,
             required_capacity,
             max_power,
             shs_price):
    """
    This function adds a node to a nodes DataFrame nodes_df:

    Parameters
    ----------
        nodes_df (pandas. DataFrame):
            Pandas DataFrame containing the nodes composing the network. In the form
                            x_coordinate  y_coordinate   required_capacity  max_power
                label
                a           0.0           0.0             3.0               3.0
                b           0.0           1.0             6.0               6.0

        x_coordinate (float):
            x coordinate of the node.

        y_coordinate (float):
            y coordinate of the node.

        required_capacity (float):
            Value of the required capacity of a shs for the node.

        max_power (float):
            Value of the maximm power required by the shs for the node.

        shs_price (float):
            Cost associated with electrify node with a solar home system.

    """

    nodes_df.at[node_label] = (x_coordinate,
                               y_coordinate,
                               required_capacity,
                               max_power,
                               shs_price)

# ---------------COMPUTE SHS PRICES ----------------#


# ---------------PROPERTIES/FEATURES OF NODES --------------#


def distance_between_nodes(node1, node2, nodes_df):
    """
        Returns the distance between two nodes of a node DataFrame.

        Parameters
        ----------
        node1: str
            Label of the first node.
        node2: str
            Label of the second node.

        nodes_df (pandas.DataFrame):
        Pandas DataFrame containing the labels and coordinates of the nodes under the form:
                    x_coordinate  y_coordinate   required_capacity  max_power
        label
        a           0.0           0.0             3.0               3.0
        b           0.0           1.0             6.0               6.0
        c           1.0           1.0             10.0              200.0
        d           2.0           0.0             2.0               2.0

        Returns
        -------
            Distance between the two nodes.
    """
    if not (node1 in nodes_df.index and node2 in nodes_df.index):
        raise Warning(f"nodes {node1} and {node2} are not in nodes_df")
        return np.infty
    return np.sqrt((nodes_df["x_coordinate"][node1]
                    - (nodes_df["x_coordinate"][node2])
                    ) ** 2
                   + (nodes_df["y_coordinate"][node1]
                      - nodes_df["y_coordinate"][node2]
                      ) ** 2
                   )


def count_number_of_connections(node_index, links_df):
    """
    This function counts the number of links connected to a node.

    Parameters
    ----------

    node_index (str):
        Index of the node.
    links_df (pandas.DataFrame)
        Pandas DataFrame containing the links connecting the network. In the form
                            node_a     node_b  distance
            label
            (node0, node1)  node0  node1    2.2360
            (node1, node2)  node0  node2    2.8284

    Output
    ------
        Number of links connecting the node.

    """
    return links_df[
        (links_df['node_a'] == node_index) | (links_df['node_b'] == node_index)
    ].shape[0]


def are_nodes_connected(node_a, node_b, links_df):
    """
    This function returns True is there exists a link in links_df connecting
    node_a and node_b.

    Parameters
    ----------
        node_a (str):
            Index of the first node.

        node_b (str):
            Index of the second node.

        links_df (pandas.DataFrame)
            Pandas DataFrame containing the links connecting the network. In the form
                            node_a   node_b  distance
            label
            node0, node1    node0  node1    2.2360
            (node1, node2)  node0  node2    2.8284
    Output
    ------
        True if there exists a link connecting node_a and node_b in links_df.
    """
    for index_link, row_link in links_df.iterrows():
        if ((row_link['node_a'] == node_a and row_link['node_b'] == node_b)
                or (row_link['node_a'] == node_b and row_link['node_b'] == node_a)
            ):
            return True
    return False


def neighoring_nodes(node_index, links_df):
    """
    This function returns a list of all the nodes that are direct neighbors of the node according to links_df.

    Parameters
    ----------
        node_index (str):
            Label of the node.

        links_df (pandas.DataFrame)
            Pandas DataFrame containing the links connecting the network. In the form
                            node_a     node_b  distance
            label
            node0, node1    node0  node1    2.2360
            (node1, node2)  node0  node2    2.8284
    Output
    ------
        List of the neighboring nodes of node_index.
    """

    neighboring_nodes = []
    for other_node in set(list(links_df['node_a']) + list(links_df['node_b'])):
        if are_nodes_connected(node_index, other_node, links_df):
            neighboring_nodes.append(other_node)
    return neighboring_nodes


def distance_to_clostest_neighbor(ref_node, links_df):
    """
    Returns the index of the clostest neighbor or the node given as parameter.

    Parameters
    ----------
    ref_node (str):
        index of the reference node considered.

    links_df (pandas.DataFrame)
            Pandas DataFrame containing the links connecting the network. In the form
                            node_a     node_b  distance
            label
            node0, node1    node0  node1    2.2360
            (node1, node2)  node0  node2    2.8284

    Output
    ------
    index of the neighboring node that is the closest to the ref_node given as parameter
    """

    return links_df[(links_df['node_a'] == ref_node) | (links_df['node_b'] == ref_node)]['distance'].min()


def nodes_on_branch(stam_node, branch_first_nodes, links_df, nodes_in_branch, iteration=0, max_iteration=np.infty):
    """
    This function recursively explores the branch of a tree and returns a list
    of all the nodes on that branch.

    Parameters
    ----------

    stam_node (str):
        Index of the node at the stamm of the branch (the node is
        not considered as part of the branch).
    branch_first_nodes (list):
        List of indices of the next nodes to be explored by the recursive
        function.
    links_df (pandas.DataFrame)
            Pandas DataFrame containing the links connecting the network.
            In the form
                            node_a     node_b  distance
            label
            node0, node1    node0  node1    2.2360
            (node1, node2)  node0  node2    2.8284
    nodes_in_branch (list):
        List of nodes already explored by the recursive function (this list
        contains the nodes identified on the branch and is completed at each
        recursion step).

    Output
    ------
        List of all the nodes on the branches originating at stam_node and
        pointing toward the nodes in branch_first_nodes.
    """
    if iteration >= max_iteration:
        return nodes_in_branch
    for branch_node in branch_first_nodes:
        if branch_node not in nodes_in_branch:
            nodes_in_branch.append(branch_node)
        if not are_nodes_connected(stam_node, branch_node, links_df):
            raise Exception(
                f'Error in excecution of nodes_on_branch, nodes '
                + f'{stam_node} and {branch_node} are not connected')

    for node in branch_first_nodes:
        neighbors = neighoring_nodes(
            node_index=node,
            links_df=links_df)
        downstream_nodes = [node for node in neighbors if (
            node != stam_node and node not in nodes_in_branch)]
        nodes_in_branch += downstream_nodes
        res = nodes_on_branch(stam_node=node,
                              branch_first_nodes=downstream_nodes,
                              links_df=links_df,
                              nodes_in_branch=nodes_in_branch,
                              iteration=iteration+1,
                              max_iteration=max_iteration)

    if len(branch_first_nodes) > 0:
        return res
    return nodes_in_branch


def betweenness_centrality(links_df):
    """
    This method returns the betweeness centrality of the nodes composing the
    network.
    It relies on the networkx.betweenness_centrality of the networkx module.
    Parameters
    ----------
        links_df (pandas.DataFrame):
            Pandas DataFrame containing the links connecting the network. In the form
                            node_a   node_b  distance
            label                                 
            node0, node1    node0  node1    2.2360
            (node1, node2)  node0  node2    2.8284
    Output
    ------
        Dictionnary with nodes indices as key and respective betweenness
        centrality measure.
    """
    graph = nx.Graph()
    for link_index, link_row in links_df.iterrows():
        graph.add_edge(link_row['node_a'], link_row['node_b'])
    return nx.betweenness_centrality(graph)

# -----------------COMPUTE MST LINKS -----------------#


def mst_links(nodes_df):
    """
    This function computes the links connecting the set of nodes so that the
    created network forms a minimum spanning tree (MST).

    Parameters
    ----------
    nodes_df (pandas.DataFrame):
        Pandas DataFrame containing the labels and coordinates of the nodes under the form:
                    x_coordinate  y_coordinate   required_capacity  max_power
        label
        a           0.0           0.0             3.0               3.0
        b           0.0           1.0             6.0               6.0
        c           1.0           1.0             10.0              200.0
        d           2.0           0.0             2.0               2.0

    Output
    ------
        Pandas Dataframe containing the (undirected) links composing the MST network.
        Example output:
                            node_a     node_b  distance
            label
            node0, node1    node0  node1    2.2360
            (node1, node2)  node0  node2    2.8284

    """
    X = np.zeros((nodes_df.shape[0], nodes_df.shape[0]))

    for i in range(nodes_df.shape[0]):
        for j in range(nodes_df.shape[0]):
            if i > j:
                index_node_i = nodes_df.index[i]
                index_node_j = nodes_df.index[j]
                X[j][i] = distance_between_nodes(
                    node1=index_node_i,
                    node2=index_node_j,
                    nodes_df=nodes_df)
    M = csr_matrix(X)

    # run minimum_spanning_tree_function
    Tcsr = minimum_spanning_tree(M)
    A = Tcsr.toarray().astype(float)

    # Create links DataFrame
    links = pd.DataFrame(
        {
            'label': pd.Series([], dtype=str),
            'node_a': pd.Series([], dtype=np.dtype(str)),
            'node_b': pd.Series([], dtype=np.dtype(str)),
            'distance': pd.Series([], dtype=np.dtype(float))
        }
    ).set_index('label')

    for i in range(len(nodes_df.index)):
        for j in range(len(nodes_df.index)):
            if i > j:
                if A[j][i] > 0:
                    links.at[f"({nodes_df.index[i]}, {nodes_df.index[j]})"] = [
                        nodes_df.index[i],
                        nodes_df.index[j],
                        distance_between_nodes(
                            nodes_df.index[i], nodes_df.index[j], nodes_df)
                    ]
    return links

# ------------------IDENTIFY NODES TO DISCONNECT FROM GRID --------------#


def nodes_to_disconnect_from_grid(nodes_df,
                                  links_df,
                                  cable_price_per_meter,
                                  additional_price_for_connection_per_node):
    """
    This function computes the nodes that would be worth assigning to a solar
    home system (shs) taking the price of the associated to a grid connection
    to a simplified minimum spanning tree network and a shs.

    The critical distance of each node is computed based on the price of shs
    associated to the capacity and max_power requirement of the node. Each link
    that is connecting a node and is larger than the critical distance of that
    node will be added to a list of links_subject_to_disconnection.

    Successively, each link from links_subject_to_disconnection is
    studied before deciding if they and the links and nodes on the downstream
    branch should be disconnected (or assigned to a shs). For each of these 
    links, the mst network is divided in two cluster written A & B (that where)
    formerly connected by that link. For each cluster, the price of connecting
    all nodes to form a mst network is compared to the price of assigning a
    shs to all nodes in the cluster and provides the saving of disconnecting
    the cluster. For each link, the cluster leading to the highest saving is
    retained.
    The link corresponding to the highest saving (if provided with shs) is
    disconnected from the grid (only for positive values of saving).

    The restulting mst network is recomputed and the algorithm iterates to
    find the next link to disconnect until there are no links to be
    disconnected with positive saving.

    Parameters
    ----------
        nodes_df (pandas.DataFrame):
            Pandas DataFrame containing the labels and coordinates of the nodes under the form:
                        x_coordinate  y_coordinate   required_capacity  max_power   shs_price
            label
            a           0.0           0.0             3.0               3.0         100
            b           0.0           1.0             6.0               6.0         200
            c           1.0           1.0             10.0              200.0       1000
            d           2.0           0.0             2.0               2.0         50

        links_df (pandas.DataFrame):
            Pandas DataFrame containing the links connecting the network.
            In the form
                            node_a     node_b  distance
            label
            node0, node1    node0  node1    2.2360
            (node1, node2)  node0  node2    2.8284

        cable_price_per_meter (float):
            Cable price per meter considered for the mst network.

        additional_price_for_connection_per_node (float):
            Additional price associated to grid connection (substracted from shs
            price in computation).

    Output
    ------
        type: list
        List of nodes to to be assigned to shs.  

    """
    if nodes_df.shape[0] == 0:
        return []
    index_of_all_nodes = list(nodes_df.index)

    nodes_shs_price = {node: nodes_df['shs_price'][node]
                       for node in nodes_df.index}

    # Create list of links subject to be disconnected to reduce price
    # These links will be identified as the ones larger than the critical
    # distance of both nodes it connects

    links_subject_to_disconnection = set()
    links_to_be_disconnected = set()
    nodes_to_be_disconnected = set()

    # Compute critical distance for each nodes and discard nodes that are
    # further away to nearest neighbor than own critical distance

    critial_distance_dict = {node: (
        nodes_shs_price[node] - additional_price_for_connection_per_node
    ) / cable_price_per_meter for node in nodes_df.index}

    for node in nodes_df.index:
        links_connecting_node_df = links_df[(
            links_df['node_a'] == node) | (links_df['node_b'] == node)]

        # Identify if nodes is further away to nearest neighbor than critical
        # distance, if so, discard it
        if (distance_to_clostest_neighbor(node, links_df) > critial_distance_dict[node]
                and (count_number_of_connections(node_index=node, links_df=links_df) == 1)):
            nodes_to_be_disconnected.add(node)
            nodes_df = nodes_df.drop(node)

            for link_to_neighbors in links_connecting_node_df.index:
                links_to_be_disconnected.add(link_to_neighbors)
                links_df = links_df.drop(link_to_neighbors)

    start_time = time.time()
    links_df = mst_links(nodes_df)

    for link_index, link_row in links_df.iterrows():
        node_a = link_row['node_a']
        node_b = link_row['node_b']

        # Remove all nodes that are further away than respective critical
        # distance from nearest neighbor. Also remove related links.

        for node in [node_a, node_b]:
            if (link_row['distance'] > critial_distance_dict[node]):
                links_subject_to_disconnection.add(link_index)

    if len(links_subject_to_disconnection) == links_df.shape[0]:
        return index_of_all_nodes
    counter = 0
    max_counter = links_df.shape[0]
    while len(links_subject_to_disconnection) > 0 and counter < max_counter:
        counter += 1

        # Compute link from links_subject_to_disconnection that is the
        # most favorable to disconnect

        most_favorable_link_to_disconnect = None
        saving_of_diconnecting_most_favorable_link = - np.infty
        nodes_disconnected_if_most_favorable_link_is_cutted = []

        links_removed_from_links_subject_to_disconnection = set()
        for link in links_subject_to_disconnection:
            if link in links_removed_from_links_subject_to_disconnection:
                break
            # Explore branch a (clsuter containing node_a and all nodes
            # downstream from a relatve to node_b)

            node_a = links_df['node_a'][link]
            node_b = links_df['node_b'][link]

            betweenness_centrality_dic = betweenness_centrality(links_df=links_df)
            betweenness_centrality_a = betweenness_centrality_dic[node_a]
            betweenness_centrality_b = betweenness_centrality_dic[node_b]
            if (betweenness_centrality_b < betweenness_centrality_a):
                node_a, node_b = node_b, node_a

            # Compute all nodes on branch a
            nodes_on_branch_a = nodes_on_branch(
                stam_node=node_b,
                branch_first_nodes=[node_a],
                links_df=links_df,
                nodes_in_branch=[])

            # Compute price of shs for all nodes in branch a
            price_shs_on_branch_a = sum([nodes_shs_price[node]
                                         for node in nodes_on_branch_a])

            # Compute links connecting the nodes of branch a ommiting other
            # links_subject_to_disconnection
            other_links_subject_to_disconnection = [
                long_link for long_link in links_subject_to_disconnection
                if long_link != link]
            links_on_branch_a_df = links_df[
                ((links_df['node_a'].isin(nodes_on_branch_a)) | (
                    links_df['node_b'].isin(nodes_on_branch_a))) &
                (~links_df.index.isin(other_links_subject_to_disconnection))]
            price_links_in_branch_a = links_on_branch_a_df['distance'].sum() * cable_price_per_meter

            price_connections_in_branch_a = len(nodes_on_branch_a) * \
                additional_price_for_connection_per_node

            # Compute saving of disconnecting branch a
            saving_of_disconecting_branch_a = (
                price_links_in_branch_a
                + price_connections_in_branch_a
                - price_shs_on_branch_a
            )

            # Compute saving of disconnecting branch b
            nodes_on_branch_b = [node for node in nodes_df.index
                                 if node not in nodes_on_branch_a]

            # Compute price of shs for all nodes in branch b
            price_shs_on_branch_b = sum([nodes_shs_price[node]
                                         for node in nodes_on_branch_b])

            # Compute links connecting the nodes of branch b
            links_on_branch_b_df = links_df[
                ((links_df['node_a'].isin(nodes_on_branch_b)) | (
                    links_df['node_b'].isin(nodes_on_branch_b))) &
                (~links_df.index.isin(other_links_subject_to_disconnection))]

            price_links_in_branch_b = links_on_branch_b_df['distance'].sum(
            ) * cable_price_per_meter

            price_connections_in_branch_b = len(nodes_on_branch_b) * \
                additional_price_for_connection_per_node

            saving_of_disconecting_branch_b = (
                price_links_in_branch_b
                + price_connections_in_branch_b
                - price_shs_on_branch_b
            )
            # Identifiy most favorable link to disconnect and nodes that
            # would thus be disconnected
            if (saving_of_disconecting_branch_a
                    > saving_of_diconnecting_most_favorable_link):

                saving_of_diconnecting_most_favorable_link =\
                    saving_of_disconecting_branch_a
                most_favorable_link_to_disconnect = link
                nodes_disconnected_if_most_favorable_link_is_cutted =\
                    nodes_on_branch_a

            if (saving_of_disconecting_branch_b
                    > saving_of_diconnecting_most_favorable_link):
                saving_of_diconnecting_most_favorable_link =\
                    saving_of_disconecting_branch_b
                most_favorable_link_to_disconnect = link
                nodes_disconnected_if_most_favorable_link_is_cutted =\
                    nodes_on_branch_b
            if max(saving_of_disconecting_branch_a, saving_of_disconecting_branch_b) < 0:
                links_removed_from_links_subject_to_disconnection.add(link)

        if saving_of_diconnecting_most_favorable_link > 0:
            # Remove most favorable link to disconnect from links_df as well as
            # all links on that branch and the nodes from nodes_df
            links_to_be_disconnected.add(
                most_favorable_link_to_disconnect)
            links_on_disconnected_branch = list(
                links_df[
                    (links_df['node_a'].isin(nodes_disconnected_if_most_favorable_link_is_cutted))
                    & (links_df['node_b'].isin(nodes_disconnected_if_most_favorable_link_is_cutted))
                ].index
            )

            links_df = links_df.drop(most_favorable_link_to_disconnect)

            links_to_be_disconnected = links_to_be_disconnected.union(links_on_disconnected_branch)
            for link_to_disconnect in links_on_disconnected_branch:
                links_df = links_df.drop(link_to_disconnect)
            nodes_to_be_disconnected = nodes_to_be_disconnected.union(
                nodes_disconnected_if_most_favorable_link_is_cutted)
            for node in nodes_disconnected_if_most_favorable_link_is_cutted:
                nodes_df = nodes_df.drop(node)
            for link_on_disconnected_branch in links_on_disconnected_branch:
                links_subject_to_disconnection.discard(link_on_disconnected_branch)
        else:
            return nodes_to_be_disconnected

        for link_retained in links_removed_from_links_subject_to_disconnection:
            links_subject_to_disconnection.discard(link_retained)
        # Remove most favorable links from links_subject_to_disconnection
        links_subject_to_disconnection.discard(most_favorable_link_to_disconnect)

        # Recompute links_df from nodes_df
        links_df = mst_links(nodes_df=nodes_df)
    return nodes_to_be_disconnected
