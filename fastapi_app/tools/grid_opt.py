from __future__ import division
import numpy as np
import pandas as pd
from k_means_constrained import KMeansConstrained
from scipy.sparse.csgraph import minimum_spanning_tree
from fastapi_app.tools.grids import Grid
from fastapi_app.tools.optimizer import Optimizer


class GridOptimizer(Optimizer):
    """
    This class includes:
        - methods for optimizing the "grid" object
        - attributes containing all default values for the optimization parameters.

    Attributes
    ----------
    ???
    """

    def __init__(
        self, start_date, n_days, project_lifetime, wacc, tax, mst_algorithm="Kruskal"
    ):
        """
        Initialize the grid optimizer object
        """
        super().__init__(start_date, n_days, project_lifetime, wacc, tax)
        self.mst_algorithm = mst_algorithm

    # ------------ CONNECT NODES USING TREE-STAR SHAPE ------------#
    def connect_grid_consumers(self, grid: Grid):
        """
        +++ ok +++

        This method create the connections between each consumer and the
        nearest pole


        Parameters
        ----------
        grid (~grids.Grid):
            grid object
        """
        links = grid.get_links()
        # Remove all existing connections between poles and consumers
        grid.clear_links(link_type="connection")

        # calculate the number of clusters and their labels obtained from kmeans clustering
        n_clusters = grid.poles()[grid.poles()["type_fixed"] == False].shape[0]
        cluster_labels = grid.poles()["cluster_label"]

        # create links between each node and the corresponding centroid
        for cluster in range(n_clusters):

            if grid.nodes[grid.nodes["cluster_label"] == cluster].index.__len__() == 1:
                continue

            # first filter the nodes and only select those with cluster labels equal to 'cluster'
            filtered_nodes = grid.nodes[grid.nodes["cluster_label"] == cluster_labels[cluster]]

            # then obtain the label of the pole which is in this cluster (as the center)
            pole_label = filtered_nodes.index[filtered_nodes["node_type"] == "pole"][0]

            for node_label in filtered_nodes.index:
                # adding consumers
                if node_label != pole_label:
                    if grid.nodes.loc[node_label, "is_connected"]:
                        grid.add_links(label_node_from=str(pole_label), label_node_to=str(node_label))
                        grid.nodes.loc[node_label, "parent"] = str(pole_label)

    def connect_grid_poles(self, grid: Grid, long_links=[]):
        """
        +++ ok +++

        This method create links between all poles together based on the
        specified algorithm for minimum spanning tree and also considering
        the added poles because of the constraint on the maximum distance
        between poles.

        Parameters
        ----------
        grid (~grids.Grid):
            grid object

        long_links (list): optional
            list of all links longer than the maximum allowed distance between
            links.
        """

        # First, all links in the grid should be removed.
        grid.clear_links(link_type="distribution")

        # Now, all links from the sparse matrix obtained using the minimum
        # spanning tree are stored in `links_mst`.
        # All poles in the `links_mst` should be connected together considering:
        #   + The number of rows in the 'links_mst' reveals the number of
        #     connections.
        #   + (x,y) of each nonzero element of the 'links_mst' correspond to the
        #     (pole_from, pole_to) labels.
        links_mst = np.argwhere(grid.grid_mst != 0)

        for link_mst in range(links_mst.shape[0]):
            mst_pole_from = grid.poles().index[links_mst[link_mst, 0]]
            mst_pole_to = grid.poles().index[links_mst[link_mst, 1]]

            # Create two different combinations for each link obtained from the
            # minimum spanning tree: (px, py) and (py, px).
            # Since the direction of the link is not important here, it is
            # assumed (px, py) = (py, px).
            mst_from_to = "(" + mst_pole_from + ", " + mst_pole_to + ")"
            mst_to_from = "(" + mst_pole_to + ", " + mst_pole_from + ")"

            # If the link obtained from the minimum spanning tree is one of the
            # long links that should be removed and replaced with smaller links,
            # this part will be executed.
            if (mst_from_to in long_links) or (mst_to_from in long_links):

                # Both `mst_from_to` and `mst_to_from` will be checked to find
                # the added poles, but only the dataframe which is not empty is
                # considered as the final `added_poles`
                added_poles_from_to = grid.poles()[
                    (grid.poles()["type_fixed"] == True)
                    & (grid.poles()["how_added"] == mst_from_to)
                ]
                added_poles_to_from = grid.poles()[
                    (grid.poles()["type_fixed"] == True)
                    & (grid.poles()["how_added"] == mst_to_from)
                ]

                # In addition to the `added_poles` a flag is defined here to
                # deal with the direction of adding additional poles.
                if not added_poles_from_to.empty:
                    added_poles = added_poles_from_to
                    to_from = False
                elif not added_poles_to_from.empty:
                    added_poles = added_poles_to_from
                    to_from = True

                # In this part, the long links are broken into smaller links.
                # `counter` represents the number of the added poles.
                counter = 0
                n_added_poles = added_poles.shape[0]
                for index_added_pole in added_poles.index:

                    if counter == 0:
                        # The first `added poles` should be connected to
                        # the beginning or to the end of the long link,
                        # depending on the `to_from` flag.
                        if to_from:
                            grid.add_links(
                                label_node_from=index_added_pole,
                                label_node_to=mst_pole_to,
                            )
                        else:
                            grid.add_links(
                                label_node_from=mst_pole_from,
                                label_node_to=index_added_pole,
                            )

                    if counter == n_added_poles - 1:
                        # The last `added poles` should be connected to
                        # the end or to the beginning of the long link,
                        # depending on the `to_from` flag.
                        if to_from:
                            grid.add_links(
                                label_node_from=mst_pole_from,
                                label_node_to=index_added_pole,
                            )
                        else:
                            grid.add_links(
                                label_node_from=index_added_pole,
                                label_node_to=mst_pole_to,
                            )

                    if counter > 0:
                        # The intermediate `added poles` should connect to
                        # the other `added_poles` before and after them.
                        grid.add_links(
                            label_node_from=added_poles.index[counter - 1],
                            label_node_to=added_poles.index[counter],
                        )
                    counter += 1

                    # Change the `how_added` tag for the new poles.
                    grid.nodes.at[index_added_pole, "how_added"] = "long-distance"

            # If `link_mst` does not belong to the list of long links, it is
            # simply connected without any further check.
            else:
                grid.add_links(
                    label_node_from=grid.poles().index[links_mst[link_mst, 0]],
                    label_node_to=grid.poles().index[links_mst[link_mst, 1]],
                )

    # ------------ MINIMUM SPANNING TREE ALGORITHM ------------ #

    def create_minimum_spanning_tree(self, grid: Grid):
        """
        Creates links between all poles using 'Prims' or 'Kruskal' algorithms
        for obtaining the minimum spanning tree.

        Parameters
        ----------
        grid (~grids.Grid):
            grid object
        """

        if self.mst_algorithm == "Prims":
            self.mst_using_prims(grid)
        elif self.mst_algorithm == "Kruskal":
            self.mst_using_kruskal(grid)
        else:
            raise Exception("Invalid value provided for mst_algorithm.")

    def mst_using_prims(self, grid: Grid):
        """
        This  method creates links between all poles following
        Prim's minimum spanning tree method. The idea goes as follow:
        a first node is selected and it is connected to the nearest neighbour,
        together they compose the a so-called forest. Then a loop
        runs over all node of the forest, the node that is the closest to
        the forest without being part of it is added to the forest and
        connected to the node of the forest it is the closest to.

        Parameters
        ----------
        grid (~grids.Grid):
            Grid object whose poles shall be connected.
        """

        # create list to keep track of nodes that have already been added
        # to the forest

        for segment in grid.get_poles()["segment"].unique():
            # Create dataframe containing the poles from the segment
            # and add temporary column to keep track of wheter the
            # pole has already been added to the forest or not
            poles = grid.get_poles()[grid.get_poles()["segment"] == segment]
            poles["in_forest"] = [False] * poles.shape[0]

            # Makes sure that there are at least two poles in segment
            if poles[-(poles["in_forest"])].shape[0] > 0:
                # First, pick one pole and add it to the forest by
                # setting its value in 'in_forest' to True
                index_first_forest_pole = poles[-poles["in_forest"]].index[0]
                poles.at[index_first_forest_pole, "in_forest"] = True

                # while there are poles not connected to the forest,
                # find nereast pole to the forest and connect it to the forest
                count = 0  # safety parameter to avoid staying stuck in loop
                while len(poles[-poles["in_forest"]]) and count < poles.shape[0]:

                    # create variables to compare poles distances and store best
                    # candidates
                    shortest_dist_to_pole_outside_forest = grid.distance_between_nodes(
                        poles[poles["in_forest"]].index[0],
                        poles[-poles["in_forest"]].index[0],
                    )
                    index_closest_pole_in_forest = poles[poles["in_forest"]].index[0]
                    index_closest_pole_to_forest = poles[-poles["in_forest"]].index[0]

                    # Iterate over all poles within the forest and over all the
                    # ones outside of the forest and find shortest distance
                    for index_pole_in_forest, row_forest_pole in poles[
                        poles["in_forest"]
                    ].iterrows():
                        for (
                            index_pole_outside_forest,
                            row_pole_outside_forest,
                        ) in poles[-poles["in_forest"]].iterrows():
                            if grid.distance_between_nodes(
                                index_pole_in_forest, index_pole_outside_forest
                            ) <= (shortest_dist_to_pole_outside_forest):
                                index_closest_pole_in_forest = index_pole_in_forest
                                index_closest_pole_to_forest = index_pole_outside_forest
                                shortest_dist_to_pole_outside_forest = (
                                    grid.distance_between_nodes(
                                        index_closest_pole_in_forest,
                                        index_closest_pole_to_forest,
                                    )
                                )
                    # create a link between pole pair
                    grid.add_links(
                        index_closest_pole_in_forest, index_closest_pole_to_forest
                    )
                    poles.at[index_closest_pole_to_forest, "in_forest"] = True
                    count += 1

    def mst_using_kruskal(self, grid: Grid):
        """
        Creates links between all poles using the Kruskal's algorithm for
        the minimum spanning tree method from scipy.sparse.csgraph.

        Parameters
        ----------
        grid (~grids.Grid):
            grid object
        """

        # total number of poles (i.e., clusters)
        poles = grid.poles()
        n_poles = poles.shape[0]

        # generate all possible edges between each pair of poles
        graph_matrix = np.zeros((n_poles, n_poles))
        for i in range(n_poles):
            for j in range(n_poles):
                # since the graph does not have a direction, only the upper part of the matrix must be filled
                if j > i:
                    graph_matrix[i, j] \
                        = grid.distance_between_nodes(label_node_1=poles.index[i], label_node_2=poles.index[j])
        # obtain the optimal links between all poles (grid_mst) and copy it in the grid object
        grid_mst = minimum_spanning_tree(graph_matrix)
        grid.grid_mst = grid_mst



    #  --------------------- K-MEANS CLUSTERING ---------------------#
    def kmeans_clustering(self, grid: Grid, n_clusters: int):
        """
        Uses a k-means clustering algorithm and returns the coordinates of the centroids.

        Pamameters
        ----------
            grid (~grids.Grid):
                grid object
            n_cluster (int):
                number of clusters (i.e., k-value) for the k-means clustering algorithm

        Return
        ------
            coord_centroids: numpy.ndarray
                A numpy array containing the coordinates of the cluster centeroids.
                Suppose there are two cluster with centers at (x1, y1) & (x2, y2),
                then the output arroy would look like:
                    array([
                        [x1, y1],
                        [x2 , y2]
                        ])
        """

        # first, all poles must be removed from the nodes list
        grid.clear_poles()
        grid_consumers = grid.get_grid_consumers()

        # gets (x,y) coordinates of all nodes in the grid
        nodes_coord = np.array([[grid_consumers.x.loc[index], grid_consumers.y.loc[index]]
                                for index in grid_consumers.index if grid_consumers.is_connected.loc[index] == True])

        # features, true_labels = make_blobs(
        #    n_samples=200,
        #    centers=3,
        #    cluster_std=2.75,
        #    random_state=42)

        # features = coord_nodes

        # call kmeans clustering with constraints (min and max number of members in each cluster )
        kmeans = KMeansConstrained(
            n_clusters=n_clusters,
            init="k-means++",  # 'k-means++' or 'random'
            n_init=10,
            max_iter=300,
            tol=1e-4,
            size_min=0,
            size_max=grid.pole_max_connection,
            random_state=0,
            n_jobs=1,
        )

        # fit clusters to the data
        kmeans.fit(nodes_coord)

        # coordinates of the centroids of the clusters
        grid_consumers["cluster_label"] = kmeans.predict(nodes_coord)
        poles = pd.DataFrame(kmeans.cluster_centers_, columns=["x", "y"])
        poles.index.name = 'cluster_label'
        poles = poles.reset_index(drop=False)
        poles.index = 'p-' + poles.index.astype(str)
        poles['node_type'] = "pole"
        poles['consumer_type'] = "n.a."
        poles['consumer_detail'] = "n.a."
        poles['is_connected'] = True
        poles['how_added'] = "k-means"
        poles['latitude'] = 0
        poles['longitude'] = 0
        poles['distance_to_load_center'] = 0
        poles['type_fixed'] = False
        poles['n_connection_links'] = "0"
        poles['n_distribution_links'] = 0
        poles['parent'] = "unknown"
        poles['distribution_cost'] = 0
        grid.nodes = pd.concat([grid_consumers, poles, grid.get_shs_consumers()], axis=0)

        # compute (lon,lat) coordinates for the poles
        grid.convert_lonlat_xy(inverse=True)


    def determine_poles(self, grid: Grid, min_n_clusters, power_house_consumers, power_house):
        """
        Computes the cost of grid based on the configuration obtained from
        the k-means clustering algorithm for different numbers of poles, and
        returns the number of poles corresponding to the lowest cost.

        Parameters
        ----------
        grid (~grids.Grid):
            'grid' object which was defined before
        min_n_clusters: int
            the minimum number of clusters required for the grid to satisfy
            the maximum number of pole connections criteria

        Return
        ------
        number_of_poles: int
            the number of polse corresponding to the minimum cost of the grid
        """

        # obtain the location of poles using kmeans clustering method
        self.kmeans_clustering(grid=grid, n_clusters=min_n_clusters)
        # create the minimum spanning tree to obtain the optimal links between poles
        if power_house is not None:
            cluster_label = grid.nodes.loc['100000', 'cluster_label']
            power_house_idx = grid.nodes[(grid.nodes["node_type"] == "pole") &
                                         (grid.nodes["cluster_label"] == cluster_label)].index
            power_house_consumers['cluster_label'] = cluster_label
            power_house_consumers['consumer_type'] = np.nan
            grid.nodes = pd.concat([grid.nodes, power_house_consumers],)
            grid.placeholder_consumers_for_power_house(remove=True)

        self.create_minimum_spanning_tree(grid)

        # connect all links in the grid based on the previous calculations
        self.connect_grid_consumers(grid)
        self.connect_grid_poles(grid)
        if power_house is not None:
            grid.nodes.loc[grid.nodes.index == power_house_idx[0], "node_type"] = 'power-house'
            grid.nodes.loc[grid.nodes.index == power_house_idx[0], "how_added"] = 'manual'


    def find_opt_number_of_poles(self, grid, connection_cable_max_length, n_mg_consumers):
        # calculate the minimum number of poles based on the
        # maximum number of connections at each pole
        if grid.pole_max_connection == 0:
            min_number_of_poles = 1
        else:
            min_number_of_poles = int(np.ceil(n_mg_consumers / (grid.pole_max_connection)))

        space = pd.Series(range(min_number_of_poles, n_mg_consumers, 1))

        def is_enough_poles(n):
            self.kmeans_clustering(grid=grid, n_clusters=n)
            self.connect_grid_consumers(grid)
            constraints_violation = grid.links[grid.links["link_type"] == "connection"]
            constraints_violation = constraints_violation[
                constraints_violation["length"] > connection_cable_max_length]
            if constraints_violation.shape[0] > 0:
                return False
            else:
                return True

        for _ in range(min_number_of_poles, n_mg_consumers, 1):
            if len(space) >= 5:
                next_n = int(space.median())
                if is_enough_poles(next_n) is True:
                    space = space[space <= next_n]
                else:
                    space = space[space > next_n]
            else:
                for next_n in space:
                    next_n = int(next_n)
                    if next_n == space.iat[-1] or is_enough_poles(next_n) is True:
                        return next_n