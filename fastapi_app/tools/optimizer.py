import numpy as np
import pandas as pd
import os
import time
import json
from k_means_constrained import KMeansConstrained
from munkres import Munkres
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.datasets import make_blobs
from sklearn.metrics import precision_recall_curve

from fastapi_app.tools.io import make_folder
from fastapi_app.tools.grids import Grid

import oemof.solph as solph
from datetime import datetime, timedelta
import pyomo.environ as po


class Optimizer:
    """
    This is a general parent class for both grid and energy system optimizers
    """

    def __init__(
        self, start_date="2021-01-01", n_days=365, project_lifetime=20, wacc=0.1, tax=0
    ):
        """
        Initialize the grid optimizer object
        """
        self.start_date = start_date
        self.n_days = n_days
        self.project_lifetime = project_lifetime
        self.wacc = wacc
        self.tax = tax

        self.crf = (self.wacc * (1 + self.wacc) ** self.project_lifetime) / (
            (1 + self.wacc) ** self.project_lifetime - 1
        )

    def capex_multi_investment(self, capex_0, component_lifetime):
        """
        Calculates the equivalent CAPEX for components with lifetime less than the project lifetime.

        """
        # convert the string type into the float type for both inputs
        capex_0 = float(capex_0)
        component_lifetime = float(component_lifetime)

        if self.project_lifetime == component_lifetime:
            number_of_investments = 1
        else:
            number_of_investments = int(
                round(self.project_lifetime / component_lifetime + 0.5)
            )

        first_time_investment = capex_0 * (1 + self.tax)

        for count_of_replacements in range(0, number_of_investments):
            if count_of_replacements == 0:
                capex = first_time_investment
            else:
                if count_of_replacements * component_lifetime != self.project_lifetime:
                    capex = capex + first_time_investment / (
                        (1 + self.wacc) ** (count_of_replacements * component_lifetime)
                    )

        # Substraction of component value at end of life with last replacement (= number_of_investments - 1)
        # This part calculates the salvage costs
        if number_of_investments * component_lifetime > self.project_lifetime:
            last_investment = first_time_investment / (
                (1 + self.wacc) ** ((number_of_investments - 1) * component_lifetime)
            )
            linear_depreciation_last_investment = last_investment / component_lifetime
            capex = capex - linear_depreciation_last_investment * (
                number_of_investments * component_lifetime - self.project_lifetime
            ) / ((1 + self.wacc) ** (self.project_lifetime))

        return capex


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

    def connect_grid_elements(self, grid: Grid):
        """
        +++ ok +++

        This method create links between:
            - each consumer and the nearest pole
            - all poles together based on the specified algorithm for minimum spanning tree

        Parameters
        ----------
        grid (~grids.Grid):
            grid object
        """

        # ===== FIRST PART: CONNECT CONSUMERS TO POLES =====

        # first, all links of the grid must be deleted
        grid.clear_links()

        # calculate the number of clusters and their labels obtained from kmeans clustering
        n_clusters = grid.poles().shape[0]
        cluster_labels = grid.poles()["cluster_label"]

        # create links between each node and the corresponding centroid
        for cluster in range(n_clusters):
            # first filter the nodes and only select those with cluster labels equal to 'cluster'
            filtered_nodes = grid.nodes[
                grid.nodes["cluster_label"] == cluster_labels[cluster]
            ]

            # then obtain the label of the pole which is in this cluster (as the center)
            pole_label = filtered_nodes.index[filtered_nodes["node_type"] == "pole"][0]

            for node_label in filtered_nodes.index:
                # adding consumers
                if node_label != pole_label:
                    grid.add_links(label_node_from=pole_label, label_node_to=node_label)

        # ===== SECOND PART: CONNECT POLES TO EACH OTHER =====

        # get all links from the sparse matrix obtained using the minimum spanning tree
        links = np.argwhere(grid.grid_mst != 0)

        # connect the poles considering the followings:
        #   + the number of rows of the 'links' reveals the number of connections
        #   + (x,y) of each nonzero element of the 'links' correspond to the (pole_from, pole_to) labels
        for link in range(links.shape[0]):
            label_pole_from = grid.poles().index[links[link, 0]]
            label_pole_to = grid.poles().index[links[link, 1]]
            grid.add_links(label_node_from=label_pole_from, label_node_to=label_pole_to)

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
        the minimum spanning tree method from scpicy.sparse.csgraph.

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
                # since the graph does not have a direction, only the upper part of the matrix must be flled
                if j > i:
                    graph_matrix[i, j] = grid.distance_between_nodes(
                        label_node_1=poles.index[i], label_node_2=poles.index[j]
                    )

        # obtain the optimal links between all poles (grid_mst) and copy it in the grid object
        grid_mst = minimum_spanning_tree(graph_matrix)
        grid.grid_mst = grid_mst

    # ------------------- ALLOCATION ALGORITHMS -------------------#

    def connect_consumer_to_nereast_poles(self, grid: Grid):
        """
        This method create a link between each consumer
        and the nereast pole of the same segment.

        Parameters
        ----------
        grid (~grids.Grid):
            Grid object.
        """

        # Iterate over all segment containing poles
        for segment in grid.get_poles()["segment"].unique():
            # Iterate over all consumers and connect each of them
            # to the closest pole or powerhub in the segment
            for index_node, row_node in grid.consumers()[
                grid.consumers()["segment"] == segment
            ].iterrows():
                # This variable is a temporary variable that is used to find
                # the nearest meter pole to a node
                index_closest_pole = grid.get_poles()[
                    grid.get_poles()["segment"] == segment
                ].index[0]
                shortest_dist_to_pole = grid.distance_between_nodes(
                    index_node, index_closest_pole
                )
                for index_pole, row_pole in grid.get_poles()[
                    grid.get_poles()["segment"] == segment
                ].iterrows():
                    # Store which pole is the clostest and what the
                    # distance to it is
                    if (
                        grid.distance_between_nodes(index_node, index_pole)
                        < shortest_dist_to_pole
                    ):
                        shortest_dist_to_pole = grid.distance_between_nodes(
                            index_node, index_pole
                        )
                        index_closest_pole = index_pole
                # Finally add the link to the grid
                grid.add_links(index_node, index_closest_pole)

    def connect_consumer_to_capacitated_poles(self, grid: Grid):
        """
        This method assigns each consumer of a grid to a pole
        of the same segment taking into consideration the maximum
        capacity of the pole and minimizing the overall distribution
        line length. It is based on the Munkres algorithm from the munkres
        module.

        Parameters
        ----------
        grid (~grids.Grid):
            Grid object.
        """

        for segment in grid.get_poles()["segment"].unique():
            poles_in_segment = grid.get_poles()[grid.get_poles()["segment"] == segment]
            consumers_in_segment = grid.consumers()[
                grid.consumers()["segment"] == segment
            ]
            num_consumers = consumers_in_segment.shape[0]
            segment_pole_capacity = grid.get_segment_pole_capacity(segment)

            # if the pole capacity is too strong of a constraint to connect
            # all the consumers to the grid
            if segment_pole_capacity < num_consumers:
                raise Exception(
                    "pole capacity only allows "
                    + str(segment_pole_capacity)
                    + " consumers to be connected to poles of segment "
                    + str(segment)
                    + ", but there are "
                    + str(num_consumers)
                    + " consumers in the segment"
                )
            else:
                # Create matrix containing the distances between the
                # consumers and the hus
                distance_matrix = []
                index_list = []
                for pole in poles_in_segment.index:
                    for allocation_slot in range(
                        grid.get_poles()["allocation_capacity"][pole]
                    ):
                        distance_list = [
                            grid.distance_between_nodes(pole, consumer)
                            for consumer in consumers_in_segment.index
                        ]
                        distance_list.extend(
                            [0] * (segment_pole_capacity - num_consumers)
                        )
                        distance_matrix.append(distance_list)
                        index_list.append(pole)
                # Call munkres_sol function for solveing allocation problem
                munkres_sol = Munkres()
                indices = munkres_sol.compute(distance_matrix)
                # Add corresponding links to the grid
                for x in indices:
                    if x[1] < consumers_in_segment.shape[0]:
                        grid.add_links(
                            index_list[x[0]], consumers_in_segment.index[int(x[1])]
                        )

    # --------------------- SEGMENTATION ---------------------#

    def propagate_segment_to_neighbours(self, grid: Grid, index, segment):
        """
        This method is a helping function used to split a segment into two.
        It is a recursice function that sets the segment of a node to a given
        value and then does the same for all of it's neighbours that have a
        different segment index. The recursion starts at the node corresponding
        to the index given as parameter.

        Parameters
        ----------
        grid (~grids.Grid):
            Grid object.

        index (str):
            index of the node the function should change the segment index.

        segment (str):
            index (label) of the segment to be set for the nodes.

        """
        for index_neighbour in grid.links()[(grid.links()["from"] == index)]["to"]:
            if not grid.get_nodes()["segment"][index_neighbour] == segment:
                grid.set_segment(index_neighbour, segment)
                self.propagate_segment_to_neighbours(grid, index_neighbour, segment)
        for index_neighbour in grid.links()[(grid.links()["to"] == index)]["from"]:
            if not grid.get_nodes()["segment"][index_neighbour] == segment:
                grid.set_segment(index_neighbour, segment)
                self.propagate_segment_to_neighbours(grid, index_neighbour, segment)

    def split_segment(self, grid: Grid, segment, min_segment_size):
        """
        This method splits a grid segment into two segments of size at least
        min_segment_size.

        Parameters
        ----------
        grid (~grids.Grid):
            Grid object.

        segment (str):
            Index of the segment to be split.

        min_segment_size (int):
            Minimum allowed size of the two parts of the segments after split.

        Notes
        -----
            This method performs temporary modifications to the grid
            nodes in order to connect all the nodes of the initial segment
            with a minimum spanning tree to see what is the longest node from
            the tree that can be removed, thus splitting the segment into two
            sub-segments of respective size at least min_segment_size. Once the
            segmentation is identified, the nodes are restored to their initial
            state and the segmentation is performed by setting the 'segment'
            property of the node to the ones identified earlier
        """
        # make sure that segment index matches a node segment
        if segment not in grid.get_nodes()["segment"].unique():
            raise Warning("the segment index doesn't correspond to any grid segment")
            return

        # make sure that the initial segment is big enough to be split into
        # two subsegments of size at least min_segment_size
        if (
            grid.get_nodes()[grid.get_nodes()["segment"] == segment].shape[0]
            < 2 * min_segment_size
        ):
            return

        # Store grid's nodes in dataframe since the actual grid is modified
        # during method
        node_backup = grid.get_nodes()
        # filter nodes to keep only the ones belongging to the initial segment
        grid.set_nodes(grid.get_nodes()[grid.get_nodes()["segment"] == segment])
        # changes all nodes into poles
        grid.set_all_node_type_to_poles()
        # Connect the nodes using MST
        grid.clear_links()

        self.create_minimum_spanning_tree(grid)

        # Create list containing links index sorted by link's distance
        index_link_sorted_by_distance = [
            index
            for index in grid.links()["distance"].nlargest(grid.links().shape[0]).index
        ]
        # Try to split the segment removing the longest link and see if
        # resulting sub-segments meet the minimum size criterion, if not,
        # try with next links (the ones just smaller) until criterion meet

        for link in index_link_sorted_by_distance:
            index_node_from = grid.links()["from"][link]
            index_node_to = grid.links()["to"][link]
            old_segment = grid.get_nodes()["segment"][index_node_from]
            segment_1 = grid.get_nodes()["segment"][index_node_from]
            segment_2 = grid.get_nodes()["segment"][index_node_to] + "_2"
            grid.set_segment(index_node_from, segment_2)
            grid.set_segment(index_node_to, segment_2)

            self.propagate_segment_to_neighbours(grid, index_node_to, segment_2)
            grid.set_segment(index_node_from, segment_1)

            if (
                grid.get_nodes()[grid.get_nodes()["segment"] == segment_1].shape[0]
                >= min_segment_size
                and grid.get_nodes()[grid.get_nodes()["segment"] == segment_2].shape[0]
                >= min_segment_size
            ):
                break
            else:
                for node_label in grid.get_nodes().index:
                    grid.set_segment(node_label, old_segment)

        segment_dict = {
            index: grid.get_nodes()["segment"][index]
            for index in grid.get_nodes().index
        }

        grid.set_nodes(node_backup)
        for index in segment_dict:
            grid.set_segment(index, segment_dict[index])

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

        # gets (x,y) coordinates of all nodes in the grid
        nodes_coord = np.array(
            [
                [grid.nodes.x.loc[index], grid.nodes.y.loc[index]]
                for index in grid.nodes.index
            ]
        )

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
        )

        # fit clusters to the data
        kmeans.fit(nodes_coord)

        # coordinates of the centroids of the clusters
        centroids_coord = kmeans.cluster_centers_

        # add the obtained centroids as poles to the grid
        counter = 0
        for i in range(n_clusters):
            centroids_label = f"p-{counter}"
            grid.add_node(
                label=centroids_label,
                x=centroids_coord[i, 0],
                y=centroids_coord[i, 1],
                node_type="pole",
                consumer_type="n.a.",
                consumer_detail="n.a.",
                is_connected=True,
                how_added="nr-optimization",
                cluster_label=counter,
            )
            counter += 1

        # compute (lon,lat) coordinates for the poles
        grid.convert_lonlat_xy(inverse=True)

        # connect different elements of the grid
        #   + consumers to the nearest poles
        #   + poles together

        # this parameter shows the label of the associated cluster to each node
        nodes_cluster_labels = kmeans.predict(nodes_coord)
        for node_label in grid.consumers().index:
            grid.nodes.cluster_label.loc[node_label] = nodes_cluster_labels[
                int(node_label)
            ]

    def find_opt_number_of_poles(self, grid: Grid, min_n_clusters: int):
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


        Notes
        -----
        The method assumes that the price of the configuration obtained using
        the k-means clustering algorithm as a function of the price is a
        striclty concave function. The method explores the prices associated
        with different number in both increasing and decreasing direction and
        stop 3 steps after the last minimum found.
        """

        # make a copy of the grid and perform changes so that the grid remains unchanged
        # grid_copy = copy.deepcopy(grid)

        # in case the grid contains 'poles' from the previous optimization
        # they must be removed, becasue the grid_optimizer will calculate
        # new locations for poles considering the newly added nodes
        # for node_with_fixed_type in grid_copy.get_nodes().index:
        #    grid_copy.set_type_fixed(node_with_fixed_type, False)
        # grid_copy.set_all_node_type_to_consumers()

        # a pandas dataframe representing the grid costs for different configurations
        grid_costs = pd.DataFrame()

        # start computing the grid costs for each case
        #   + starting number of poles: minimum number of required poles
        number_of_nodes = grid.nodes.shape[0]

        # obtain the location of poles using kmeans clustering method
        self.kmeans_clustering(grid=grid, n_clusters=min_n_clusters)

        # create the minimum spanning tree to obtain the optimal links between poles
        self.create_minimum_spanning_tree(grid)

        # connect all links in the grid based on the previous calculations
        self.connect_grid_elements(grid)

        # after the kmeans clustering algorithm, the  number of poles is obtained
        number_of_poles = grid.poles().shape[0]

        # initialize dataframe to store number of poles and corresponding costs
        grid_costs = pd.DataFrame({"poles": [], "cost": []}).set_index("poles")

        # put the first row of this dataframe
        grid_costs.loc[number_of_poles] = grid.cost()

        # create variable that is used to compare cost of the grid
        # corresponding to different number of poles
        compare_cost = grid_costs.cost.min()

        # initialize counter and iteration numbers to limit exploration of solution space
        # and to stop searching when costs are increasing after n steps (n = stop_counter)
        counter = 0
        iteration = 0
        stop_counter = 3

        while (
            ((grid.cost() >= compare_cost) or (counter < stop_counter))
            and (number_of_poles < int(number_of_nodes * 0.8))
            and (iteration < number_of_nodes)
        ):
            iteration += 1

            # increase the number of poles and let the kmeans clustering algorithm re-calculate
            # all new locations for poles, and the minimum spanning tree find the new interpole links
            number_of_poles += 1
            self.kmeans_clustering(grid=grid, n_clusters=number_of_poles)

            # create the minimum spanning tree to obtain the optimal links between poles
            self.create_minimum_spanning_tree(grid)

            # connect all links in the grid based on the previous calculations
            self.connect_grid_elements(grid)

            # obtain the new cost for the new configuration of the grid
            grid_costs.loc[number_of_poles] = grid.cost()

            # if grid cost increases, the counter will also increase
            if grid.cost() >= compare_cost:
                counter += 1
            else:
                counter = 0

            compare_cost = grid_costs.cost.min()

        # update the grid object based on the optimum number of the poles
        opt_number_of_poles = int(grid_costs.index[np.argmin(grid_costs)])
        self.kmeans_clustering(grid=grid, n_clusters=opt_number_of_poles)
        self.create_minimum_spanning_tree(grid)
        self.connect_grid_elements(grid)

        return opt_number_of_poles

    # -----------------------REMOVE NODE-------------------------#

    def remove_last_node(self, grid: Grid):
        """
        Removes the last node added to a grid.

        Parameters
        ----------
        grid (~grids.Grid):
            Grid object
        """
        if grid.get_nodes().shape[0] > 0:
            grid.set_nodes(grid.get_nodes().drop(grid.get_nodes().index[-1]))
            if not grid.is_pole_capacity_constraint_too_strong():
                self.connect_grid_elements(grid)

    # ---------- MAKE GRID COMPLIANT WITH CAPACITY CONSTRAINTS --------- #

    def flip_consumers_until_pole_capacity_constraint_met(self, grid: Grid):
        """
        This method is ment to be called when the pole capacity constraint
        is too restrictive for the nodes. The method flips the node type of
        consumers until there are enough poles in each segment to meet the pole
        capacity constraint.

        Returns
        ---------
        """

        if grid.get_default_pole_capacity() > 0:

            if grid.get_poles().shape[0] == 0:
                grid.flip_random_node()

            if grid.is_pole_capacity_constraint_too_strong():
                for segment in grid.get_nodes()["segment"].unique():
                    while grid.consumers()[
                        grid.consumers()["segment"] == segment
                    ].shape[0] > grid.get_segment_pole_capacity(segment):
                        random_number = np.random.rand()
                        num_consumers = grid.consumers()[
                            grid.consumers()["segment"] == segment
                        ].shape[0]
                        grid.flip_node(
                            grid.consumers()[grid.consumers() == segment].index[
                                int(random_number * num_consumers)
                            ]
                        )

    # --------------- NETWORK RELAXATION METHODS --------------- #

    def nr_optimization(
        self,
        grid,
        number_of_poles: int,
        number_of_relaxation_steps: int,
        damping_factor: float = 0.5,
        weight_of_attraction: str = "constant",
        first_guess_strategy="random",
        number_of_steps_bewteen_random_shifts: int = 0,
        number_of_hill_climbers_runs: int = 0,
        save_output: bool = False,
        output_folder: str = None,
        print_progress_bar: bool = True,
    ):
        """
        This iterative process finds an approximate solution for the minimum cost of the grid.
        The main idea is to consider the links between poles as pulling strings exercing a force on each pole.
        To this end, virtual poles are added to the grid, which are free to locate wherever
        on the plane. At each iteration step, they are shifted in the direction of
        the resulting "strength" from the nodes they are linked with.

        Parameters
        ----------
            grid (~grids.Grid):
                grid object

            number_of_poles (int):
                number of poles in the grid

            number_of_relaxation_steps (int):
                number of iteration in the relaxation process

            damping_factor (int):
                a factor determining how much the virtual poles can be shifted
                together with shortest distance between pair of nodes

            weight_of_attraction (str):
                defines how strong each link attracts/pulls the pole.
                Possibilites are:
                    + 'constant':
                        only depends on the type of link.
                    + 'distance':
                        depends on the type of the link and is proportional to the
                        distance of the link. (i.e., long links pull stronger).

            first_guess_strategy (str):
                the stategy that should be used to get the starting configuration
                Possibilites are:
                    + 'k_means': (default)
                        virtual poles are initially located at center of cluster
                        obtained using a k_means clustering algorithm from the
                        sklearn library
                    + 'random':
                        virtual nodes are randomly located in box containing
                        all grid nodes
                    + 'relax_input_grid':
                        the starting configuration is the grid configuration given as input

            number_of_steps_bewteen_random_shifts (int):
                determines how often a random shift of one of the poles should occur

            save_output (bool):
                determines whether or not the output grid and the log should be
                saved in the output_folder.

            number_of_hill_climbers_runs (int):
                when larger than 0, local cost optimization is performed
                after the relaxation. The local optimization process
                computes in which direction each pole should be shifted in
                order to reduce the grid costs. The process is repeated
                'number_of_hill_climbers_runs' times

            output_folder (str):
                path of the folder the grid output of the algorithm should be
                saved in.

            print_progress_bar (bool):
                determines whether or not the progress bar should be displayed
                in the console.

        Output
        ------
            class:`pandas.core.frame.DataFrame`
                log dataframe containg the detail of the run as well as the,
                time evolution, the 'virtual_cost' (see notes for more info)
                evolution and a measure of how much the poles are shifted at
                each step.

        TODO: check if we need this or not

        Notes
        -----
            The 'virtual_cost' is the cost of the grid containing the poles.
            Since, during the process, the layout is
            not a feasible solution (the virtual poles are not located at house
            location), the price that is computed cannot be interpreted as the
            price of a feasible grid layout
        """

        if save_output:
            # show the summary of problem in the terminal
            print(f"|{42 * '_'}| NETWORK RELAXATION |{42 * '_'}|\n")
            print(f"{35 * ' '}number of poles:{8 * ' '} {number_of_poles}\n")
            print(
                f"{35 * ' '}number of steps:{8 * ' '}" + f"{number_of_relaxation_steps}"
            )
            print("\n")
            print(
                f"{35 * ' '}first guess strategy:{3 * ' '}" + f"{first_guess_strategy}"
            )
            print(
                f"\n{35 * ' '}weight of attraction:{3 * ' '}"
                + f"{weight_of_attraction}\n\n"
            )

            # save all results in a folder
            if output_folder is None:
                path_to_folder = f"data/output/{grid.get_id()}"
            else:
                path_to_folder = output_folder
            make_folder(path_to_folder)

            folder_name = (
                f"{grid.get_id()}_{number_of_poles}_poles_"
                + f"{number_of_relaxation_steps}_steps_"
                + f"attr-{weight_of_attraction[:4]}"
            )

            if os.path.exists(f"{path_to_folder}/{folder_name}"):
                counter = 1
                while os.path.exists(f"{path_to_folder}/{folder_name}_{counter}"):
                    counter += 1
                folder_name_with_path = f"{path_to_folder}/{folder_name}_{counter}"
                folder_name = f"{folder_name}_{counter}"
                make_folder(folder_name_with_path)
            else:
                folder_name_with_path = f"{path_to_folder}/{folder_name}"
                make_folder(folder_name_with_path)

        # find out the range of (x,y) coordinate for all nodes of the grid
        x_range = [grid.nodes.x.min(), grid.nodes.x.max()]
        y_range = [grid.nodes.y.min(), grid.nodes.y.max()]

        # create log dataframe that will store info about run
        info_log = pd.DataFrame(
            {
                "time": pd.Series([0] * number_of_relaxation_steps, dtype=float),
                "cost": pd.Series([0] * number_of_relaxation_steps, dtype=float),
                "norm_longest_shift": pd.Series(
                    [0] * number_of_relaxation_steps, dtype=float
                ),
            }
        )

        """
        # FIXME: I think the first guess strategy can be deleted
        # Define number of virtual poles
        number_of_virtual_poles = (number_of_poles
                                   - grid_copy.get_poles()[
                                       grid_copy.get_poles()[
                                           'type_fixed']].shape[0]
                                   )
        # TODO: must be checked later. Maybe there is no need for that
        if first_guess_strategy == 'random':
            # flip all non-fixed poles from the grid for the optimization
            for pole, row in grid_copy.get_poles().iterrows():
                if not row['type_fixed']:
                    grid_copy.flip_node(pole)

            # Create virtual poles and add them randomly at locations within
            # square containing all nodes
            for i in range(number_of_virtual_poles):
                grid_copy.add_node(label=f'V{i}',
                                   x_coordinate=random.uniform(x_range[0],
                                                               x_range[1]),
                                   y_coordinate=random.uniform(y_range[0],
                                                               y_range[1]),
                                   node_type='pole',
                                   segment='0',
                                   allocation_capacity=allocation_capacity)

        elif first_guess_strategy == 'k_means':
            # TODO: I guess the flipping part should be removed because here we have poles AND consumers
            # flip all non-fixed poles from the grid for the optimization
            for pole, row in grid_copy.get_poles().iterrows():
                if not row['type_fixed']:
                    grid_copy.flip_node(pole)

            # Create virtual poles and add them at centers of clusters
            # given by the k_means_cluster_centers() method

            cluster_centers = self.kmeans_clustering(
                grid=grid_copy,
                n_clusters=number_of_virtual_poles)

            count = 0
            for coord in cluster_centers:
                grid_copy.add_node(label=f'V{count}',
                                   x_coordinate=int(coord[0]),
                                   y_coordinate=int(coord[1]),
                                   node_type='pole',
                                   segment='0',
                                   allocation_capacity=allocation_capacity)
                count += 1

        elif first_guess_strategy == 'relax_input_grid':
            counter = 0
            intial_pole_indices = grid_copy.get_poles().index

            for pole in intial_pole_indices:
                grid_copy.add_node(
                    label=f'V{counter}',
                    x_coordinate=grid_copy.get_poles()['x_coordinate'][pole],
                    y_coordinate=grid_copy.get_poles()['y_coordinate'][pole],
                    node_type='pole',
                    type_fixed=False,
                    segment='0',
                    allocation_capacity=allocation_capacity)
                grid_copy.set_node_type(pole, 'consumer')
                counter += 1

        else:
            raise Warning("invalid first_guess_strategy parameter, "
                          + "possibilities are:\n- 'random'\n- 'k_means'\n- "
                          + "'relax_input_grid'")
        self.connect_grid_elements(grid_copy)
        """
        start_time = time.time()

        # ---------- STEP 0 - Initialization step -------- #
        if print_progress_bar:
            self.print_progress_bar(0, 1)

        # Compute the new relaxation_df
        relaxation_df = self.nr_compute_relaxation_df(grid)

        # Store the 'weighted_vector' from the current and the previous steps
        # to adapt the 'damping_factor'.
        # At each step, the scalar product between the 'weighted_vector'
        # form the previous and current step will be computed and used
        # to adapt the 'damping_factor' value
        weighted_vectors_previous_step = relaxation_df["weighted_vector"]
        # cost_connection = grid.get_price_consumer()

        # Compute the 'damping_factor' such that the norm of the 'weighted_vector'
        # at step 0 is equal to n% of the smallest link in the grid.
        # 'n' is specified when calling nr_optimization.
        # The the 'damping_factor' is applied to the 'weighted_vector' for each pole
        for pole in relaxation_df.index:
            multiplier = (
                damping_factor
                * self.nr_smallest_link(grid)
                / self.nr_max_length_weighted_vector(relaxation_df)
            )
            relaxation_df.weighted_vector.loc[pole] = np.multiply(
                relaxation_df.weighted_vector.loc[pole], multiplier
            )

        # Shift poles in the direction of the 'weighted_vector'
        for pole in grid.poles().index:
            grid.shift_node(
                node=pole,
                delta_x=relaxation_df.weighted_vector[pole][0],
                delta_y=relaxation_df.weighted_vector[pole][1],
            )

        # update the (lon,lat) coordinates based on the new (x,y) coordinates for poles
        grid.convert_lonlat_xy(inverse=True)

        # create a new network based on the new coordinates of the poles in the grid
        self.connect_grid_elements(grid)

        # update the log file
        info_log["time"][0] = time.time() - start_time
        info_log["cost"][0] = grid.cost()
        info_log["norm_longest_shift"][0] = self.nr_max_length_weighted_vector(
            relaxation_df
        ) / self.nr_smallest_link(grid)
        if print_progress_bar:
            self.print_progress_bar(
                iteration=1, total=number_of_relaxation_steps + 1, cost=grid.cost()
            )

        # ------------ STEP n + 1 - ITERATIVE STEP ------------- #
        for n in range(1, number_of_relaxation_steps + 1):
            # Compute the new relaxation_df after the previous changes.
            relaxation_df = self.nr_compute_relaxation_df(grid)

            # Update the current 'weighted_vector'
            weighted_vectors_current_step = relaxation_df["weighted_vector"]

            # For each pole, compute the scalar product of the 'weighted_vector'
            # from the previous and the current step.
            # The values will be used to adapt the damping value
            scalar_product_weighted_vectors = [
                x1[0] * x2[0] + x1[1] * x2[1]
                for x1, x2 in zip(
                    weighted_vectors_current_step, weighted_vectors_previous_step
                )
            ]
            if min(scalar_product_weighted_vectors) >= 0:
                damping_factor = damping_factor * 2.5
            else:
                damping_factor = damping_factor / 1.5

            # Apply the 'damping_factor' to the 'weighted_vector' for each pole.
            for pole in relaxation_df.index:
                multiplier = (
                    damping_factor
                    * self.nr_smallest_link(grid)
                    / self.nr_max_length_weighted_vector(relaxation_df)
                )
                relaxation_df.weighted_vector.loc[pole] = np.multiply(
                    relaxation_df.weighted_vector.loc[pole], multiplier
                )

            # Shift poles in the direction of the 'weighted_vector'.
            for pole in grid.poles().index:
                grid.shift_node(
                    node=pole,
                    delta_x=relaxation_df.weighted_vector[pole][0],
                    delta_y=relaxation_df.weighted_vector[pole][1],
                )

            # Update the (lon,lat) coordinates based on the new (x,y) coordinates for poles.
            grid.convert_lonlat_xy(inverse=True)

            # Create a new network based on the new coordinates of the poles in the grid.
            self.connect_grid_elements(grid)

            # Update the log file.
            info_log["time"][n] = time.time() - start_time
            info_log["cost"][n] = grid.cost()
            info_log["norm_longest_shift"][n] = self.nr_max_length_weighted_vector(
                relaxation_df
            ) / self.nr_smallest_link(grid)

            # Update the progress bar.
            if print_progress_bar:
                self.print_progress_bar(
                    iteration=n + 1,
                    total=number_of_relaxation_steps + 1,
                    cost=grid.cost(),
                )

            weighted_vectors_previous_step = weighted_vectors_current_step

        # if number_of_hill_climbers_runs is non-zero, perform hill climber
        # runs
        if number_of_hill_climbers_runs > 0 and print_progress_bar:
            print("\n\nHill climber runs...\n")
        for i in range(number_of_hill_climbers_runs):
            if print_progress_bar:
                current_price = grid.price()
                self.print_progress_bar(
                    iteration=i, total=number_of_hill_climbers_runs, cost=current_price
                )

            counter = 0
            for pole in grid.poles().index:
                counter += 1
                gradient = self.nr_compute_local_price_gradient(grid, pole)
                self.nr_shift_pole_toward_minus_gradient(
                    grid=grid_copy, pole=pole, gradient=gradient
                )
                self.connect_grid_elements(grid_copy)
                if save_output:
                    info_log.loc[f"{info_log.shape[0]}"] = [
                        time.time() - start_time,
                        grid_copy.price() - (number_of_virtual_poles * cost_connection),
                        0,
                    ]
                if print_progress_bar:
                    current_price = grid_copy.price() - (
                        number_of_virtual_poles * cost_connection
                    )
                    self.print_progress_bar(
                        iteration=i + ((counter + 1) / grid_copy.get_poles().shape[0]),
                        total=number_of_hill_climbers_runs,
                        cost=current_price,
                    )

        n_final = number_of_relaxation_steps + 1
        info_log["time"][n_final] = time.time() - start_time
        info_log["cost"][n_final] = grid.cost()
        info_log["norm_longest_shift"][n_final] = self.nr_max_length_weighted_vector(
            relaxation_df
        ) / self.nr_smallest_link(grid)

        if save_output:
            print(f"\n\nFinal price: {grid_copy.price()} $\n")

            info_log.to_csv(path_to_folder + "/" + folder_name + "/log.csv")
            grid_copy.export(
                backup_name=folder_name,
                folder=path_to_folder,
                allow_saving_in_existing_backup_folder=True,
            )

            # create json file containing about dictionary
            about_dict = {
                "grid id": grid_copy.get_id(),
                "number_of_poles": number_of_poles,
                "number_of_relaxation_steps": number_of_relaxation_steps,
                "damping_factor": damping_factor,
                "weight_of_attraction": weight_of_attraction,
                "first_guess_strategy": first_guess_strategy,
                "number_of_steps_bewteen_random_shifts": number_of_steps_bewteen_random_shifts,
                "number_of_hill_climbers_runs": number_of_hill_climbers_runs,
                "save_output": save_output,
                "output_folder": output_folder,
                "print_progress_bar": print_progress_bar,
            }

            json.dumps(about_dict)

            with open(
                path_to_folder + "/" + folder_name + "/about_run.json", "w"
            ) as about:
                about.write(json.dumps(about_dict))

    def nr_smallest_link(self, grid: Grid):
        """
        +++ ok +++

        This method returns the length of the smallest link in the grid.

        Parameter
        ---------
            grid (~grids.Grid): Grid object.

        Return
        ------
            float: length of the smallest link in the grid
        """
        return min([x for x in grid.links.length if x > 0])

    def nr_max_length_weighted_vector(self, relaxation_df):
        """
        +++ ok +++

        This method returns the norm (i.e. magnitude) of the longest 'weighted_vector'
        from the relaxation DataFrame.

        Parameter
        ---------
            relaxation_df : :class:`pandas.core.frame.DataFrame`
                DataFrame containing all relaxation vectors for each invidiuda
                poles.

        Return
        ------
            float
                Norm of the longest vector in vector_resulting.
        """

        max_length = 0.0

        for pole in relaxation_df.index:
            x_weighted_vector = relaxation_df.weighted_vector.loc[pole][0]
            y_weighted_vector = relaxation_df.weighted_vector.loc[pole][1]
            norm_weighted_vector = np.sqrt(
                x_weighted_vector**2 + y_weighted_vector**2
            )
            if norm_weighted_vector > max_length:
                max_length = norm_weighted_vector

        return max_length

    def nr_get_smaller_distance_between_nodes(self, grid: Grid):
        """
        This methods returns the distance between the two nodes
        from the grid that are the closest.

        Parameter
        ---------
        grid (~grids.Grid):
            Grid object.

        Output
        ------
            float
                distance between the two nodes in [m] from the grid that are
                the closest.
        """

        smaller_distance = grid.distance_between_nodes(
            grid.get_nodes().index[0], grid.get_nodes().index[1]
        )

        node_indices = grid.consumers().index
        for i in range(len(node_indices)):
            for j in range(len(node_indices)):
                if i > j:
                    if (
                        grid.distance_between_nodes(node_indices[i], node_indices[j])
                        < smaller_distance
                    ):
                        smaller_distance = grid.distance_between_nodes(
                            node_indices[i], node_indices[j]
                        )
        return smaller_distance

    def nr_compute_relaxation_df(self, grid: Grid):
        """
        +++ ok +++

        This method computes the vectors between all poles and nodes
        that are connected to it. The Series 'weighted_vector' is the
        sum of the vector multiplied by the corresponding costs per cable
        length (i.e., distribution and interpole cabes).

        Parameters
        ----------
            grid (~grids.Grid):
                Grid object.

        Return
        ------
            class:`pandas.core.frame.DataFrame`
                DataFrame containing all relaxation vectors for each invidiudal
                poles and the final equivalent vector called 'weighted_vector'
        """

        # Get costs per meter of cables.
        epc_hv_cable = grid.epc_hv_cable
        epc_lv_cable = grid.epc_lv_cable

        # A dataframe including all data required for relaxation.
        relaxation_df = pd.DataFrame(
            {
                "pole": pd.Series([], dtype=str),
                "connected_consumers": pd.Series([]),
                "xy_each_consumer": pd.Series([]),
                "xy_equivalent_consumers": pd.Series([]),
                "connected_poles": pd.Series([]),
                "xy_each_pole": pd.Series([]),
                "xy_equivalent_poles": pd.Series([]),
                "weighted_vector": pd.Series([]),
            }
        ).set_index("pole")

        # Update the 'relaxation_df' for all poles in the grid object.
        for pole in grid.poles().index:
            relaxation_df.loc[pole] = [[], [], [0, 0], [], [], [0, 0], [0, 0]]

            # Find the connected poles and consumers to each pole
            for link in grid.links.index:

                # Links labels are strings in '(from, to)' format. So, first both
                # parentheses and the comma must be removed to get 'from' and 'to' separately.
                link_from = (link.replace("(", "")).replace(")", "").split(", ")[0]
                link_to = (link.replace("(", "")).replace(")", "").split(", ")[1]

                if link_from == pole:
                    if link_to in grid.poles().index:
                        relaxation_df["connected_poles"][pole].append(link_to)
                    elif link_to in grid.consumers().index:
                        relaxation_df["connected_consumers"][pole].append(link_to)

                elif link_to == pole:
                    if link_from in grid.poles().index:
                        relaxation_df["connected_poles"][pole].append(link_from)
                    elif link_from in grid.consumers().index:
                        relaxation_df["connected_consumers"][pole].append(link_from)

            # Calculate the relative (x,y) positions of poles and consumers connected to
            # the pole being investigated.
            for consumer in relaxation_df["connected_consumers"][pole]:
                delta_x = grid.nodes.x[consumer] - grid.nodes.x[pole]
                delta_y = grid.nodes.y[consumer] - grid.nodes.y[pole]
                relaxation_df["xy_each_consumer"][pole].append([delta_x, delta_y])
                relaxation_df["xy_equivalent_consumers"][pole][0] += delta_x
                relaxation_df["xy_equivalent_consumers"][pole][1] += delta_y
                relaxation_df["weighted_vector"][pole][0] += (
                    delta_x * epc_lv_cable / epc_hv_cable
                )
                relaxation_df["weighted_vector"][pole][1] += (
                    delta_y * epc_lv_cable / epc_hv_cable
                )

            for pole_2 in relaxation_df["connected_poles"][pole]:
                delta_x = grid.nodes.x[pole_2] - grid.nodes.x[pole]
                delta_y = grid.nodes.y[pole_2] - grid.nodes.y[pole]
                relaxation_df["xy_each_pole"][pole].append([delta_x, delta_y])
                relaxation_df["xy_equivalent_poles"][pole][0] += delta_x
                relaxation_df["xy_equivalent_poles"][pole][1] += delta_y
                relaxation_df["weighted_vector"][pole][0] += delta_x
                relaxation_df["weighted_vector"][pole][1] += delta_y

        return relaxation_df

    def nr_compute_local_price_gradient(self, grid: Grid, pole, delta=1):
        """
        This method computes the price of four neighboring configurations
        obatined by shifting the pole given as input by delta respectively
        in the positive x direction, in the negative x directtion, in the
        positive y direction and in the negative y direction. The gradient
        vector is computed as follow:
        (f(x + d, y) * e_x - f(x - d, y) * e_x
        + f(x, y + d) * e_y - f(x, y -d) * e_y)/d
        where d = delta, e_x and e_y are respectively the unit vectors in
        direction x and y and f(x, y) is the price of the grid with the pole
        given as input located at (x, y).

        Parameters
        ----------
        grid (~grids.Grid):
            Grid object.

        pole (str):
            index of the pole whose gradient is computed.

        delta: float
            Pixel distance used for computing the gradient.
        """

        # compute price of configuration with pole shifted from (delta, 0)
        grid.shift_node(node=pole, delta_x=delta, delta_y=0)
        self.connect_grid_elements(grid)
        price_pos_x = grid.cost()

        # compute price of configuration with pole shifted from (- delta, 0)
        grid.shift_node(node=pole, delta_x=-2 * delta, delta_y=0)
        self.connect_grid_elements(grid)
        price_neg_x = grid.cost()

        # compute price of configuration with pole shifted from (0, delta)
        grid.shift_node(node=pole, delta_x=delta, delta_y=delta)
        self.connect_grid_elements(grid)
        price_pos_y = grid.cost()

        # compute price of configuration with pole shifted from (0, - delta)
        grid.shift_node(node=pole, delta_x=0, delta_y=-2 * delta)
        self.connect_grid_elements(grid)
        price_neg_y = grid.cost()

        # Shift pole back to initial position
        grid.shift_node(node=pole, delta_x=0, delta_y=delta)
        self.connect_grid_elements(grid)

        gradient = (
            (price_pos_x - price_neg_x) / delta,
            (price_pos_y - price_neg_y) / delta,
        )

        return gradient

    def nr_shift_pole_toward_minus_gradient(self, grid: Grid, pole, gradient):
        """
        This method compares the price of the grid if pole is shifted by
        different amplitudes toward the negative gradient direction and
        performs shift that result in better price improvement.

        Parameters
        ----------
        grid (~grids.Grid):
            Grid object.

        pole (str):
            Index of the pole whose gradient is computed:

        gradient: tuple
            Two-dimensional vector pointing in price gradient direction.
        """

        # Store initial coordinates of pole to be shifted
        nodes = grid.get_nodes()
        links = grid.links()

        amplitude = 15
        price_after_shift = grid.cost()
        price_before_shift = grid.cost()

        counter = 0
        while price_after_shift <= price_before_shift and counter < 20:
            nodes = grid.get_nodes()
            links = grid.links()
            price_before_shift = grid.cost()
            grid.shift_node(pole, -amplitude * gradient[0], -amplitude * gradient[1])
            self.connect_grid_elements(grid)
            amplitude *= 3
            counter += 1

            price_after_shift = grid.cost()
        grid.set_nodes(nodes)
        grid.set_links(links)
        self.connect_grid_elements(grid)

    def print_progress_bar(
        self,
        iteration: int,
        total: int,
        prefix: str = "",
        suffix: str = "",
        decimals: int = 1,
        length: int = 50,
        fill: str = "█",
        print_end: str = "\r",
        cost: float = None,
    ):
        """
        +++ ok +++

        Call in a loop to create terminal progress bar.

        Parameters
        ----------
            iteration   - Required  : current iteration (int)
            total       - Required  : total iterations (int)
            prefix      - Optional  : prefix string (str)
            suffix      - Optional  : suffix string (str)
            decimals    - Optional  : positive number of decimals in percent
                                    complete (int)
            length      - Optional  : character length of bar (int)
            fill        - Optional  : bar fill character (str)
            print_end   - Optional  : end character (e.g., "\r", "\r\n") (str)

            Notes
            -----
                Funtion inspired from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/30740258 # noqa: E501
        """
        # the precentage which is shown next to the progress bar
        precision = "{:." + str(decimals) + "f}"
        percent = precision.format(100 * iteration / total)

        # fill some part of the progress bar depending on the iteration number
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + "-" * (length - filled_length)

        # update the progress bar based on the iteration number and cost
        if cost is None:
            print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=print_end)
        else:
            print(
                f"\r{prefix} |{bar}| {percent}% {suffix}, Cost: {cost:.2f} USD",
                end=print_end,
            )

        # print a new empty line after all iterations
        if iteration == total:
            print()

    def display_progress_bar(self, current, final, message=""):
        """
        This method displays a progress bar on the console. The progress is
        displayed in percent and corresponds to current/final. The message
        parameter is appended  after the progress bar.

        Parameters
        ----------
        current: float
            Current iteration.

        final: float
            Final iteration.

        message (str):
            Diplayed after the progress bar.

        """
        if current > final:
            final = current

        current_in_percent = int(current / final * 50)
        remaining = 50 - current_in_percent

        bar = "█" * current_in_percent + "-" * remaining
        print(f"\r|{bar}|  {int(current / final * 100)}%   {message}", end="")


class EnergySystemOptimizer(Optimizer):
    """
    This class includes:
        - methods for optimizing the "energy system" object
        - attributes containing all default values for the optimization parameters.

    Attributes
    ----------
    ???
    """

    def __init__(
        self,
        start_date,
        n_days,
        project_lifetime,
        wacc,
        tax,
        path_data="",
        solver="cbc",
        pv={
            "settings": {"is_selected": True, "design": True},
            "parameters": {
                "nominal_capacity": None,
                "capex": 1000,
                "opex": 20,
                "lifetime": 20,
            },
        },
        diesel_genset={
            "settings": {"is_selected": True, "design": True},
            "parameters": {
                "nominal_capacity": None,
                "capex": 1000,
                "opex": 20,
                "variable_cost": 0.045,
                "lifetime": 8,
                "fuel_cost": 1.214,
                "fuel_lhv": 11.83,
                "min_load": 0.3,
                "max_efficiency": 0.3,
            },
        },
        battery={
            "settings": {"is_selected": True, "design": True},
            "parameters": {
                "nominal_capacity": None,
                "capex": 350,
                "opex": 7,
                "lifetime": 6,
                "soc_min": 0.3,
                "soc_max": 1,
                "c_rate_in": 1,
                "c_rate_out": 0.5,
                "efficiency": 0.8,
            },
        },
        inverter={
            "settings": {"is_selected": True, "design": True},
            "parameters": {
                "nominal_capacity": None,
                "capex": 400,
                "opex": 8,
                "lifetime": 10,
                "efficiency": 0.98,
            },
        },
        rectifier={
            "settings": {"is_selected": True, "design": True},
            "parameters": {
                "nominal_capacity": None,
                "capex": 400,
                "opex": 8,
                "lifetime": 10,
                "efficiency": 0.98,
            },
        },
        shortage={
            "settings": {"is_selected": True},
            "parameters": {
                "max_total": 10,
                "max_timestep": 50,
                "penalty_cost": 0.3,
            },
        },
    ):
        """
        Initialize the grid optimizer object
        """
        super().__init__(start_date, n_days, project_lifetime, wacc, tax)
        self.path_data = path_data
        self.solver = solver
        self.pv = pv
        self.diesel_genset = diesel_genset
        self.battery = battery
        self.inverter = inverter
        self.rectifier = rectifier
        self.shortage = shortage

    def create_datetime_objects(self):
        """
        explanation
        """
        start_date_obj = datetime.strptime(self.start_date, "%Y-%m-%d")
        self.start_date = start_date_obj.date()
        self.start_time = start_date_obj.time()
        self.start_datetime = datetime.combine(
            start_date_obj.date(), start_date_obj.time()
        )
        # conversion to in() is needed becasue self.n_days is a <class 'numpy.int64'> and it causes troubles
        self.end_datetime = self.start_datetime + timedelta(days=int(self.n_days))

    def import_data(self):
        data = pd.read_csv(filepath_or_buffer=self.path_data)
        data.index = pd.date_range(
            start=self.start_datetime, periods=len(data), freq="H"
        )

        self.solar_potential = data.SolarGen.loc[
            self.start_datetime: self.end_datetime
        ]
        self.demand = data.Demand.loc[self.start_datetime: self.end_datetime]
        self.solar_potential_peak = self.solar_potential.max()
        self.demand_peak = self.demand.max()

    def optimize_energy_system(self):
        self.create_datetime_objects()
        self.import_data()

        # define an empty dictionary for all epc values
        self.epc = {}
        date_time_index = pd.date_range(
            start=self.start_date, periods=self.n_days * 24, freq="H"
        )
        energy_system = solph.EnergySystem(timeindex=date_time_index)

        # -------------------- BUSES --------------------
        # create electricity and fuel buses
        b_el_ac = solph.Bus(label="electricity_ac")
        b_el_dc = solph.Bus(label="electricity_dc")
        b_fuel = solph.Bus(label="fuel")

        # -------------------- SOURCES --------------------
        # fuel density is assumed 0.846 kg/l
        fuel_cost = (
            self.diesel_genset["parameters"]["fuel_cost"]
            / 0.846
            / self.diesel_genset["parameters"]["fuel_lhv"]
        )
        fuel_source = solph.Source(
            label="fuel_source", outputs={b_fuel: solph.Flow(variable_costs=fuel_cost)}
        )

        self.epc["pv"] = (
            self.crf
            * self.capex_multi_investment(
                capex_0=self.pv["parameters"]["capex"],
                component_lifetime=self.pv["parameters"]["lifetime"],
            )
            + self.pv["parameters"]["opex"]
        )
        # Make decision about optimization strategy of the PV
        if self.pv["settings"]["design"] == True:
            # DESIGN
            pv = solph.Source(
                label="pv",
                outputs={
                    b_el_dc: solph.Flow(
                        fix=self.solar_potential / self.solar_potential_peak,
                        nominal_value=None,
                        investment=solph.Investment(
                            ep_costs=self.epc["pv"] * self.n_days / 365
                        ),
                        variable_costs=0,
                    )
                },
            )
        else:
            # DISPATCH
            pv = solph.Source(
                label="pv",
                outputs={
                    b_el_dc: solph.Flow(
                        fix=self.solar_potential / self.solar_potential_peak,
                        nominal_value=self.pv["parameters"]["nominal_capacity"],
                        variable_costs=0,
                    )
                },
            )

        # -------------------- TRANSFORMERS --------------------
        # optimize capacity of the fuel generator
        self.epc["diesel_genset"] = (
            self.crf
            * self.capex_multi_investment(
                capex_0=self.diesel_genset["parameters"]["capex"],
                component_lifetime=self.diesel_genset["parameters"]["lifetime"],
            )
            + self.diesel_genset["parameters"]["opex"]
        )

        if self.diesel_genset["settings"]["design"] == True:
            # DESIGN
            diesel_genset = solph.Transformer(
                label="diesel_genset",
                inputs={b_fuel: solph.Flow()},
                outputs={
                    b_el_ac: solph.Flow(
                        nominal_value=None,
                        variable_costs=self.diesel_genset["parameters"][
                            "variable_cost"
                        ],
                        investment=solph.Investment(
                            ep_costs=self.epc["diesel_genset"] * self.n_days / 365
                        ),
                    )
                },
                conversion_factors={
                    b_el_ac: self.diesel_genset["parameters"]["max_efficiency"]
                },
            )
        else:
            # DISPATCH
            diesel_genset = solph.Transformer(
                label="diesel_genset",
                inputs={b_fuel: solph.Flow()},
                outputs={
                    b_el_ac: solph.Flow(
                        nominal_value=self.diesel_genset["parameters"][
                            "nominal_capacity"
                        ],
                        variable_costs=self.diesel_genset["parameters"][
                            "variable_cost"
                        ],
                    )
                },
                conversion_factors={
                    b_el_ac: self.diesel_genset["parameters"]["max_efficiency"]
                },
            )

        self.epc["rectifier"] = (
            self.crf
            * self.capex_multi_investment(
                capex_0=self.rectifier["parameters"]["capex"],
                component_lifetime=self.rectifier["parameters"]["lifetime"],
            )
            + self.rectifier["parameters"]["opex"]
        )
        if self.rectifier["settings"]["design"] == True:
            # DESIGN
            rectifier = solph.Transformer(
                label="rectifier",
                inputs={
                    b_el_ac: solph.Flow(
                        nominal_value=None,
                        investment=solph.Investment(
                            ep_costs=self.epc["rectifier"] * self.n_days / 365
                        ),
                        variable_costs=0,
                    )
                },
                outputs={b_el_dc: solph.Flow()},
                conversion_factor={
                    b_el_dc: self.rectifier["parameters"]["efficiency"],
                },
            )
        else:
            # DISPATCH
            rectifier = solph.Transformer(
                label="rectifier",
                inputs={
                    b_el_ac: solph.Flow(
                        nominal_value=self.rectifier["parameters"]["nominal_capacity"],
                        variable_costs=0,
                    )
                },
                outputs={b_el_dc: solph.Flow()},
                conversion_factor={
                    b_el_dc: self.rectifier["parameters"]["efficiency"],
                },
            )

        self.epc["inverter"] = (
            self.crf
            * self.capex_multi_investment(
                capex_0=self.inverter["parameters"]["capex"],
                component_lifetime=self.inverter["parameters"]["lifetime"],
            )
            + self.inverter["parameters"]["opex"]
        )
        if self.inverter["settings"]["design"] == True:
            # DESIGN
            inverter = solph.Transformer(
                label="inverter",
                inputs={
                    b_el_dc: solph.Flow(
                        nominal_value=None,
                        investment=solph.Investment(
                            ep_costs=self.epc["inverter"] * self.n_days / 365
                        ),
                        variable_costs=0,
                    )
                },
                outputs={b_el_ac: solph.Flow()},
                conversion_factor={
                    b_el_ac: self.inverter["parameters"]["efficiency"],
                },
            )
        else:
            # DISPATCH
            inverter = solph.Transformer(
                label="inverter",
                inputs={
                    b_el_dc: solph.Flow(
                        nominal_value=self.inverter["parameters"]["nominal_capacity"],
                        variable_costs=0,
                    )
                },
                outputs={b_el_ac: solph.Flow()},
                conversion_factor={
                    b_el_ac: self.inverter["parameters"]["efficiency"],
                },
            )
        # -------------------- battery --------------------
        self.epc["battery"] = (
            self.crf
            * self.capex_multi_investment(
                capex_0=self.battery["parameters"]["capex"],
                component_lifetime=self.battery["parameters"]["lifetime"],
            )
            + self.battery["parameters"]["opex"]
        )
        if self.battery["settings"]["design"] == True:
            # DESIGN
            battery = solph.GenericStorage(
                label="battery",
                nominal_storage_capacity=None,
                investment=solph.Investment(
                    ep_costs=self.epc["battery"] * self.n_days / 365
                ),
                inputs={b_el_dc: solph.Flow(variable_costs=0)},
                outputs={b_el_dc: solph.Flow(investment=solph.Investment(ep_costs=0))},
                initial_storage_capacity=0.0,
                min_storage_level=self.battery["parameters"]["soc_min"],
                max_storage_level=self.battery["parameters"]["soc_max"],
                balanced=True,
                inflow_conversion_factor=self.battery["parameters"]["efficiency"],
                outflow_conversion_factor=self.battery["parameters"]["efficiency"],
                invest_relation_input_capacity=self.battery["parameters"]["c_rate_in"],
                invest_relation_output_capacity=self.battery["parameters"][
                    "c_rate_out"
                ],
            )
        else:
            # DISPATCH
            battery = solph.GenericStorage(
                label="battery",
                nominal_storage_capacity=self.battery["parameters"]["nominal_capacity"],
                inputs={b_el_dc: solph.Flow(variable_costs=0)},
                outputs={b_el_dc: solph.Flow()},
                initial_storage_capacity=0.0,
                min_storage_level=self.battery["parameters"]["soc_min"],
                max_storage_level=self.battery["parameters"]["soc_max"],
                balanced=True,
                inflow_conversion_factor=self.battery["parameters"]["efficiency"],
                outflow_conversion_factor=self.battery["parameters"]["efficiency"],
                invest_relation_input_capacity=self.battery["parameters"]["c_rate_in"],
                invest_relation_output_capacity=self.battery["parameters"][
                    "c_rate_out"
                ],
            )

        # -------------------- SINKS --------------------
        demand_el = solph.Sink(
            label="electricity_demand",
            inputs={
                b_el_ac: solph.Flow(
                    # min=1-max_shortage_timestep,
                    fix=self.demand / self.demand_peak,
                    nominal_value=self.demand_peak
                )
            },
        )

        excess_sink = solph.Sink(
            label="excess_sink",
            inputs={b_el_ac: solph.Flow()},
        )

        # add all objects to the energy system
        energy_system.add(
            pv,
            fuel_source,
            b_el_dc,
            b_el_ac,
            b_fuel,
            inverter,
            rectifier,
            diesel_genset,
            battery,
            demand_el,
            excess_sink,
        )

        # -------------------- SHORTAGE --------------------
        # maximal unserved demand and the variable costs of unserved demand.
        if self.shortage["settings"]["is_selected"]:
            shortage = solph.Source(
                label="shortage",
                outputs={
                    b_el_ac: solph.Flow(
                        variable_costs=self.shortage["parameters"]["shortage_penalty_cost"],
                        nominal_value=self.shortage["parameters"]["max_shortage_total"] *
                        sum(self.demand),
                        summed_max=1,
                    ),
                },
            )
        else:
            shortage = solph.Source(
                label="shortage",
                outputs={
                    b_el_ac: solph.Flow(
                        nominal_value=0,
                    ),
                },
            )
        energy_system.add(shortage)

        model = solph.Model(energy_system)

        def shortage_per_timestep_rule(model, t):
            expr = 0
            ## ------- Get demand at t ------- #
            demand = model.flow[b_el_ac, demand_el, t]
            expr += self.shortage["parameters"]["max_shortage_timestep"] * demand
            ## ------- Get shortage at t------- #
            expr += -model.flow[shortage, b_el_ac, t]

            return expr >= 0

        if self.shortage["settings"]["is_selected"]:
            model.shortage_timestep = po.Constraint(
                model.TIMESTEPS, rule=shortage_per_timestep_rule
            )

        def max_excess_electricity_total_rule(model):
            max_excess_electricity = 0.05  # fraction
            expr = 0
            ## ------- Get generated at t ------- #
            generated_diesel_genset = sum(model.flow[diesel_genset, b_el_ac, :])
            generated_pv = sum(model.flow[inverter, b_el_ac, :])
            ac_to_dc = sum(model.flow[b_el_ac, rectifier, :])
            generated = generated_diesel_genset + generated_pv - ac_to_dc
            expr += (generated * max_excess_electricity)
            ## ------- Get excess at t------- #
            excess = sum(model.flow[b_el_ac, excess_sink, :])
            expr += -excess

            return expr >= 0

        model.max_excess_electricity_total = po.Constraint(
            rule=max_excess_electricity_total_rule
        )

        # optimize the energy system
        # gurobi --> 'MipGap': '0.01'
        # cbc --> 'ratioGap': '0.01'
        solver_option = {"gurobi": {"MipGap": "0.03"}, "cbc": {"ratioGap": "0.03"}}

        model.solve(
            solver=self.solver,
            solve_kwargs={"tee": True},
            cmdline_options=solver_option[self.solver],
        )
        energy_system.results["meta"] = solph.processing.meta_results(model)
        self.results_main = solph.processing.results(model)

        self.process_results()

    def process_results(self):
        results_pv = solph.views.node(results=self.results_main, node="pv")
        results_fuel_source = solph.views.node(
            results=self.results_main, node="fuel_source"
        )
        results_diesel_genset = solph.views.node(
            results=self.results_main, node="diesel_genset"
        )
        results_inverter = solph.views.node(results=self.results_main, node="inverter")
        results_rectifier = solph.views.node(
            results=self.results_main, node="rectifier"
        )
        results_battery = solph.views.node(results=self.results_main, node="battery")
        results_demand_el = solph.views.node(
            results=self.results_main, node="electricity_demand"
        )
        results_excess_sink = solph.views.node(
            results=self.results_main, node="excess_sink"
        )
        results_shortage = solph.views.node(
            results=self.results_main, node="shortage"
        )

        # -------------------- SEQUENCES (DYNAMIC) --------------------
        # hourly demand profile
        self.sequences_demand = results_demand_el["sequences"][
            (("electricity_ac", "electricity_demand"), "flow")
        ]

        # hourly profiles for solar potential and pv production
        self.sequences_pv = results_pv["sequences"][(("pv", "electricity_dc"), "flow")]

        # hourly profiles for fuel consumption and electricity production in the fuel genset
        # the 'flow' from oemof is in kWh and must be converted to liter
        self.sequences_fuel_consumption = (
            results_fuel_source["sequences"][(("fuel_source", "fuel"), "flow")]
            / self.diesel_genset["parameters"]["fuel_lhv"]
            / 0.846
        )  # conversion: kWh -> kg -> l

        self.sequences_genset = results_diesel_genset["sequences"][
            (("diesel_genset", "electricity_ac"), "flow")
        ]

        # hourly profiles for charge, discharge, and content of the battery
        self.sequences_battery_charge = results_battery["sequences"][
            (("electricity_dc", "battery"), "flow")
        ]

        self.sequences_battery_discharge = results_battery["sequences"][
            (("battery", "electricity_dc"), "flow")
        ]

        self.sequences_battery_content = results_battery["sequences"][
            (("battery", "None"), "storage_content")
        ]

        # hourly profiles for inverted electricity from dc to ac
        self.sequences_inverter = results_inverter["sequences"][
            (("inverter", "electricity_ac"), "flow")
        ]

        # hourly profiles for inverted electricity from ac to dc
        self.sequences_rectifier = results_rectifier["sequences"][
            (("rectifier", "electricity_dc"), "flow")
        ]

        # hourly profiles for excess ac and dc electricity production
        self.sequences_excess = results_excess_sink["sequences"][
            (("electricity_ac", "excess_sink"), "flow")
        ]

        # hourly profiles for shortages in the demand coverage
        self.sequences_shortage = results_shortage["sequences"][
            (("shortage", "electricity_ac"), "flow")
        ]

        # -------------------- SCALARS (STATIC) --------------------
        if self.diesel_genset["settings"]["design"] == True:
            self.capacity_genset = results_diesel_genset["scalars"][
                (("diesel_genset", "electricity_ac"), "invest")
            ]
        else:
            self.capacity_genset = self.diesel_genset["parameters"]["nominal_capacity"]

        if self.pv["settings"]["design"] == True:
            self.capacity_pv = results_pv["scalars"][
                (("pv", "electricity_dc"), "invest")
            ]
        else:
            self.capacity_pv = self.pv["parameters"]["nominal_capacity"]

        if self.inverter["settings"]["design"] == True:
            self.capacity_inverter = results_inverter["scalars"][
                (("electricity_dc", "inverter"), "invest")
            ]
        else:
            self.capacity_inverter = self.inverter["parameters"]["nominal_capacity"]

        if self.rectifier["settings"]["design"] == True:
            self.capacity_rectifier = results_rectifier["scalars"][
                (("electricity_ac", "rectifier"), "invest")
            ]
        else:
            self.capacity_rectifier = self.rectifier["parameters"]["nominal_capacity"]

        if self.battery["settings"]["design"] == True:
            self.capacity_battery = results_battery["scalars"][
                (("electricity_dc", "battery"), "invest")
            ]
        else:
            self.capacity_battery = self.battery["parameters"]["nominal_capacity"]

        self.total_renewable = (
            (
                self.epc["pv"] * self.capacity_pv
                + self.epc["inverter"] * self.capacity_inverter
                + self.epc["battery"] * self.capacity_battery
            )
            * self.n_days
            / 365
        )
        self.total_non_renewable = (
            self.epc["diesel_genset"] * self.capacity_genset
            + self.epc["rectifier"] * self.capacity_rectifier
        ) * self.n_days / 365 + self.diesel_genset["parameters"][
            "variable_cost"
        ] * self.sequences_genset.sum(
            axis=0
        )
        self.total_component = self.total_renewable + self.total_non_renewable
        self.total_fuel = self.diesel_genset["parameters"][
            "fuel_cost"
        ] * self.sequences_fuel_consumption.sum(axis=0)
        self.total_revenue = self.total_component + self.total_fuel
        self.total_demand = self.sequences_demand.sum(axis=0)
        self.lcoe = 100 * self.total_revenue / self.total_demand

        self.res = (
            100
            * self.sequences_pv.sum(axis=0)
            / (self.sequences_genset.sum(axis=0) + self.sequences_pv.sum(axis=0))
        )

        self.excess_rate = (
            100
            * self.sequences_excess.sum(axis=0)
            / (self.sequences_genset.sum(axis=0) - self.sequences_rectifier.sum(axis=0) + self.sequences_inverter.sum(axis=0))
        )
        self.genset_to_dc = (
            100
            * self.sequences_rectifier.sum(axis=0)
            / self.sequences_genset.sum(axis=0)
        )
        self.shortage = (
            100
            * self.sequences_shortage.sum(axis=0) / self.sequences_demand.sum(axis=0)
        )

        print("")
        print(40 * "*")
        print(f"LCOE:\t {self.lcoe:.2f} cent/kWh")
        print(f"RES:\t {self.res:.0f}%")
        print(f"Excess:\t {self.excess_rate:.1f}% of the total production")
        print(f"Shortage: {self.shortage:.1f}% of the total demand")
        print(f"AC--DC:\t {self.genset_to_dc:.1f}% of the genset production")
        print(40 * "*")
        print(f"genset:\t {self.capacity_genset:.0f} kW")
        print(f"pv:\t {self.capacity_pv:.0f} kW")
        print(f"st:\t {self.capacity_battery:.0f} kW")
        print(f"inv:\t {self.capacity_inverter:.0f} kW")
        print(f"rect:\t {self.capacity_rectifier:.0f} kW")
        print(f"peak:\t {self.sequences_demand.max():.0f} kW")
        print(f"excess:\t {self.sequences_excess.max():.0f} kW")
        print(40 * "*")
