import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx


def create_nodes_df():
    """
    This function creates and returns an empty DataFrame represetning nodes of
    following form:
                    x_coordinate  y_coordinate   required_capacity  max_power
        label
        a           0.0           0.0             3.0               3.0
        b           0.0           1.0             6.0               6.0
        c           1.0           1.0             10.0              200.0
        d           2.0           0.0             2.0               2.0

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
            pd.Series([], dtype=np.dtype(float))
        }
    ).set_index('label')


def add_node(nodes_df,
             node_label,
             x_coordinate,
             y_coordinate,
             required_capacity,
             max_power):
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

    """

    nodes_df.at[node_label] = (x_coordinate,
                               y_coordinate,
                               required_capacity,
                               max_power)


def shs_price_for_load(capacity, max_power, shs_characteristics):
    """
    This function returns the price of a Solar Home System of a given an
    average load value as well as a max power based on the price ranges given
    by the price_for_shs DataFrame.

    Parameters
    ----------
    capacity (float):
        Value of the battery capactity required for the shs

    max_power (float):
        Maximum power that shs is supposed to deliver

    shs_characteristics (pandasz.DataFrame):
        Dataframe where each row contains the following inforamtions about the shs:
            'price[$]'
            'capacity[Wh]'
            'max_power[W]'

    Output
    ------
        Price of the cheapest shs fullfiling the capacity and max_power requirement criteria

    """
    for index, row in shs_characteristics.sort_values(by=['capacity[Wh]']).iterrows():
        if row['capacity[Wh]'] >= capacity and row['max_power[W]'] >= max_power:
            return row['price[$]']
    return np.infty


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


def shs_price_of_node(node_index, nodes_df, shs_characteristics):
    """
    This function returns the price of the shs coresponding to the building
    represented by the node node_index of nodes_df.

    Parameters
    ----------
        node_index (str):
            Index of the node.

        nodes_df (pandas.DataFrame):
            DataFrame containing the nodes of the network in the form
                            x_coordinate  y_coordinate
                   x_coordinate  y_coordinate    required_capacity  max_power
            label
            node0  2.0           3.0             90.0               5.0
            node1  4.0           6.0             500.0              20.0
            node2  7.0           10.0            190.0              10.0
            node3  1.0           2.0             1200.0             1000.0

        shs_characteristics (pandas.DataFrame):
            Dataframe where each row contains the following inforamtions about the shs:
                'price[$]'
                'capacity[Wh]'
                'max_power[W]'

    """

    return shs_price_for_load(
        capacity=nodes_df['required_capacity'][node_index],
        max_power=nodes_df['max_power'][node_index],
        shs_characteristics=shs_characteristics
    )


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


def nodes_on_branch(stam_node, branch_first_nodes, links_df, nodes_in_branch, iteration=0, max_iteration=5):
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
    print(f"nodes_on_branch iteration, iteration: ({iteration}/{max_iteration})")
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


def nodes_and_links_to_discard_old(nodes_df,
                                   links_df,
                                   cable_price_per_meter,
                                   additional_price_for_connection_per_node,
                                   shs_characteristics):
    print("start nodes_and_links_to_discard_old")
    links_to_remove = []
    shs_nodes_selection = []

    for link_index, link_row in links_df.iterrows():

        list_nodes_in_cluster_a = nodes_on_branch(
            stam_node=link_row['node_b'],
            branch_first_nodes=[link_row['node_a']],
            links_df=links_df,
            nodes_in_branch=[])

    # Cluster A
        df_links_in_cluster_a = links_df[links_df.node_a.isin(
            list_nodes_in_cluster_a) | links_df.node_b.isin(list_nodes_in_cluster_a)]

        price_shs_in_cluster_a = 0

        for node_a in list_nodes_in_cluster_a:
            price_shs_in_cluster_a += shs_price_of_node(
                node_a, nodes_df=nodes_df, shs_characteristics=shs_characteristics)

        price_df_links_in_cluster_a = df_links_in_cluster_a.sum()[
            'distance'] * cable_price_per_meter

        cost_of_disconnecting_cluster_a = price_shs_in_cluster_a - \
            (price_df_links_in_cluster_a +
             additional_price_for_connection_per_node * len(list_nodes_in_cluster_a))

        # Cluster B

        list_nodes_in_cluster_b = [
            node for node in nodes_df.index if node not in list_nodes_in_cluster_a]

        df_links_in_cluster_b = links_df[links_df.node_b.isin(
            list_nodes_in_cluster_b) | links_df.node_b.isin(list_nodes_in_cluster_b)]

        price_shs_in_cluster_b = 0

        for node_b in list_nodes_in_cluster_b:
            price_shs_in_cluster_b += shs_price_of_node(
                node_b, nodes_df=nodes_df, shs_characteristics=shs_characteristics)

        price_df_links_in_cluster_b = df_links_in_cluster_b.sum()[
            'distance'] * cable_price_per_meter

        cost_of_disconnecting_cluster_b = price_shs_in_cluster_b - \
            (price_df_links_in_cluster_b +
             additional_price_for_connection_per_node * len(list_nodes_in_cluster_b))
        if min(cost_of_disconnecting_cluster_a, cost_of_disconnecting_cluster_b) < 0:
            if cost_of_disconnecting_cluster_a < cost_of_disconnecting_cluster_b:
                shs_nodes_selection += list_nodes_in_cluster_a
                links_to_remove += list(df_links_in_cluster_a.index)
            else:
                shs_nodes_selection += list_nodes_in_cluster_b
                links_to_remove += list(df_links_in_cluster_b.index)

    return set(shs_nodes_selection), set(links_to_remove)


def nodes_and_links_to_discard(nodes_df,
                               links_df,
                               cable_price_per_meter,
                               additional_price_for_connection_per_node,
                               shs_characteristics):
    if nodes_df.shape[0] == 0:
        return []

    print("start nodes_and_links_to_discard (v1)")
    nodes_shs_price = {node: shs_price_of_node(
        node,
        nodes_df=nodes_df,
        shs_characteristics=shs_characteristics)
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

        if distance_to_clostest_neighbor(node, links_df) > critial_distance_dict[node]:
            nodes_to_be_disconnected.add(node)
            nodes_df.drop(node)

            for link_to_neighbors in links_connecting_node_df.index:
                links_to_be_disconnected.add(link_to_neighbors)
                links_df.drop(link_to_neighbors)
            print(f"node: {node} is further away to nn than critical distance")
            print(f"links: {links_connecting_node_df.index}")

    #################
    for link_index, link_row in links_df.iterrows():
        # The algorithm will explore the cluster A (branch with stam at node_a
        # pointing outward relative to node_b).
        # If node_b is a leaf-node, switch node_a & node_b
        node_a = link_row['node_a']
        node_b = link_row['node_b']

        critial_distance_node_a = (
            nodes_shs_price[node_a] - additional_price_for_connection_per_node
        ) / cable_price_per_meter

        critial_distance_node_b = (
            nodes_shs_price[node_b] - additional_price_for_connection_per_node
        ) / cable_price_per_meter

        # First remove all nodes that are further away than respective critical
        # distance from nearest neighbor. Also remove related links.

        for node in [node_a, node_b]:
            critial_distance_dict

            distance_to_closest_neighbor_of_node = distance_to_clostest_neighbor(node, links_df)

            if ((link_row['distance'] > critial_distance_dict[node])
                    and (distance_to_closest_neighbor_of_node > critial_distance_dict[node])):
                links_to_be_disconnected.add(link_index)
                links_df.drop(link_index)
                nodes_to_be_disconnected.add(node)
                nodes_df.drop(node)
            else:
                links_subject_to_disconnection.add(link_index)

    #################

    print(
        f"\n\n####### --->>>{len(links_subject_to_disconnection)}/{links_df.shape[0]} links_subject_to_disconnection: {links_subject_to_disconnection} before starting while loop\n\n\n")
    if len(links_subject_to_disconnection) == links_df.shape[0]:
        return list(nodes_df.index)
    counter = 0
    max_counter = links_df.shape[0]
    print(f"max_counter: {max_counter}")
    while len(links_subject_to_disconnection) > 0 and counter < max_counter:
        counter += 1

        # Compute link from links_subject_to_disconnection that is the
        # most favorable to disconnect

        most_favorable_link_to_disconnect = None
        saving_of_diconnecting_most_favorable_link = - np.infty
        nodes_disconnected_if_most_favorable_link_is_cutted = []

        print(
            f"\n\nwhile loop counter {counter}\n----> {nodes_df.shape[0]} nodes in nodes_df: {nodes_df.index}\n----> {links_df.shape[0]} links in links_df: {links_df.index}")
        print(f"links subjects to disconnection: {links_subject_to_disconnection}")

        links_removed_from_links_subject_to_disconnection = set()
        for link in links_subject_to_disconnection:
            if link in links_removed_from_links_subject_to_disconnection:
                break
            print(f"\n considering link {link}")
            # Explore branch a (clsuter containing node_a and all nodes
            # downstream from a relatve to node_b)

            node_a = link_row['node_a']
            node_b = link_row['node_b']

            betweenness_centrality_dic = betweenness_centrality(links_df=links_df)
            betweenness_centrality_a = betweenness_centrality_dic[node_a]
            betweenness_centrality_b = betweenness_centrality_dic[node_b]
            if (betweenness_centrality_b < betweenness_centrality_a):
                print(
                    f"switched nodes node_a and node_b because betweenness centralities are (a,b)=({betweenness_centrality_a}, {betweenness_centrality_b})")
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
            print(
                f"saving_if branch A of {link} is disconnected: {saving_of_disconecting_branch_a}")

            print(
                f"saving_if branch B of {link} is disconnected: {saving_of_disconecting_branch_b}")

            # Identifiy most favorable link to disconnect and nodes that
            # would thus be disconnected
            loop_entered = []
            if (saving_of_disconecting_branch_a
                    > saving_of_diconnecting_most_favorable_link):

                loop_entered.append("a")
                saving_of_diconnecting_most_favorable_link =\
                    saving_of_disconecting_branch_a
                most_favorable_link_to_disconnect = link
                print(f"links on branch A to disconnect: {links_on_branch_a_df.index}")
                nodes_disconnected_if_most_favorable_link_is_cutted =\
                    nodes_on_branch_a

            if (saving_of_disconecting_branch_b
                    > saving_of_diconnecting_most_favorable_link):
                loop_entered.append("b")
                saving_of_diconnecting_most_favorable_link =\
                    saving_of_disconecting_branch_b
                most_favorable_link_to_disconnect = link
                print(f"links on branch B to disconnect: {links_on_branch_b_df.index}")
                nodes_disconnected_if_most_favorable_link_is_cutted =\
                    nodes_on_branch_b
            if max(saving_of_disconecting_branch_a, saving_of_disconecting_branch_b) < 0:
                links_removed_from_links_subject_to_disconnection.add(link)
            print(f"loop_entered: {loop_entered}")
            print(f"nodes on branch a: {nodes_on_branch_a}")
            print(f"nodes on branch b: {nodes_on_branch_b}")
            print(
                f"   loop -> nodes_disconnected_if_most_favorable_link_is_cutted: {nodes_disconnected_if_most_favorable_link_is_cutted}")
        if saving_of_diconnecting_most_favorable_link > 0:
            print(
                f"    saving_of_diconnecting_most_favorable_link for {most_favorable_link_to_disconnect} is {saving_of_diconnecting_most_favorable_link} > 0")
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
            print(f"links_on_disconnected_branch: {links_on_disconnected_branch}")

            links_df = links_df.drop(most_favorable_link_to_disconnect)

            links_to_be_disconnected = links_to_be_disconnected.union(links_on_disconnected_branch)
            print(f"small links to disconnect: {links_on_disconnected_branch}")
            for link_to_disconnect in links_on_disconnected_branch:
                print(f"    drop link {link_to_disconnect} from nodes_df")
                links_df = links_df.drop(link_to_disconnect)
            nodes_to_be_disconnected = nodes_to_be_disconnected.union(
                nodes_disconnected_if_most_favorable_link_is_cutted)
            for node in nodes_disconnected_if_most_favorable_link_is_cutted:
                print(f"    drop node {node} from nodes_df")
                nodes_df = nodes_df.drop(node)
            print(f"       remaining {links_df.shape[0]} links: {links_df.index}")
            print(f"       remaining {nodes_df.shape[0]} nodes: {nodes_df.index}")
            for link_on_disconnected_branch in links_on_disconnected_branch:
                links_subject_to_disconnection.discard(link_on_disconnected_branch)
        else:
            print(f"link: {most_favorable_link_to_disconnect} is not worth disconnecting")
            print(f"return: {nodes_to_be_disconnected}")
            return nodes_to_be_disconnected

        for link_retained in links_removed_from_links_subject_to_disconnection:
            links_subject_to_disconnection.discard(link_retained)
        # Remove most favorable links from links_subject_to_disconnection
        links_subject_to_disconnection.discard(most_favorable_link_to_disconnect)
        print(f"updated links_subject_to_disconnection: {links_subject_to_disconnection}\n\n\n")

        print(f"counter: {counter}")
        print(f"most favorable link to disconnect: {most_favorable_link_to_disconnect}")
        print(f"associated nodes: {nodes_disconnected_if_most_favorable_link_is_cutted}")
        print(
            f"\n\nend of while loop counter {counter}\n----> {nodes_df.shape[0]} nodes in nodes_df: {nodes_df.index}\n----> {links_df.shape[0]} links in links_df: {links_df.index}\n\n\n")

    print(f"return: {nodes_to_be_disconnected}")
    return nodes_to_be_disconnected


def betweenness_centrality(links_df):
    """
    This method returns the betweeness centrality of the nodes composing the
    network.
    It relies on the networkx.betweenness_centrality of the networkx module.
    Parameters
    ----------
        links_df (pandas.DataFrame)
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
