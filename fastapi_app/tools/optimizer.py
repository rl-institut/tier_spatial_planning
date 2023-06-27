from __future__ import division
import numpy as np
import pandas as pd
from k_means_constrained import KMeansConstrained
from munkres import Munkres
from scipy.sparse.csgraph import minimum_spanning_tree
from fastapi_app.tools.grids import Grid
import oemof.solph as solph
from datetime import datetime, timedelta
import pyomo.environ as po
from pyproj import Proj
from fastapi_app.io.db import sync_queries, queries_demand, models


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
        solar_potential,
        demand,
        solver="cbc",
        pv={
            "settings": {"is_selected": True,
                         "design": True},
            "parameters": {
                "nominal_capacity": None,
                "capex": 1000,
                "opex": 20,
                "lifetime": 20,
            },
        },
        diesel_genset={
            "settings": {"is_selected": True, "design": True, "offset": False},
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
        self.solver = solver
        self.pv = pv
        self.diesel_genset = diesel_genset
        self.battery = battery
        self.inverter = inverter
        self.rectifier = rectifier
        self.shortage = shortage
        self.solar_potential = solar_potential
        self.demand = demand

    def create_datetime_objects(self):
        """
        explanation
        """
        start_date_obj = self.start_date
        self.start_date = start_date_obj.date()
        self.start_time = start_date_obj.time()
        self.start_datetime = datetime.combine(start_date_obj.date(), start_date_obj.time())
        # conversion to in() is needed becasue self.n_days is a <class 'numpy.int64'> and it causes troubles
        self.end_datetime = self.start_datetime + timedelta(days=int(self.n_days))

    def import_data(self):
        # ToDo create correct timestamps
        self.demand.index = pd.date_range(start=self.start_datetime, periods=len(self.demand.index), freq='H')
        self.demand = self.demand.loc[self.start_datetime : self.end_datetime]['Demand']
        self.solar_potential_peak = self.solar_potential.max()
        self.demand_peak = self.demand.max()

    def optimize_energy_system(self):
        self.create_datetime_objects()
        self.import_data()

        # define an empty dictionary for all epc values
        self.epc = {}
        date_time_index = pd.date_range(start=self.start_date, periods=self.n_days * 24, freq="H")
        self.solar_potential = self.solar_potential[date_time_index]
        energy_system = solph.EnergySystem(timeindex=date_time_index,
                                           infer_last_interval=True)

        # -------------------- BUSES --------------------
        # create electricity and fuel buses
        b_el_ac = solph.Bus(label="electricity_ac")
        b_el_dc = solph.Bus(label="electricity_dc")
        b_fuel = solph.Bus(label="fuel")

        # -------------------- PV --------------------
        self.epc["pv"] = (
            self.crf
            * self.capex_multi_investment(
                capex_0=self.pv["parameters"]["capex"],
                component_lifetime=self.pv["parameters"]["lifetime"],
            )
            + self.pv["parameters"]["opex"]
        )
        # Make decision about different simulation modes of the PV
        if self.pv["settings"]["is_selected"] == False:
            pv = solph.components.Source(
                label="pv",
                outputs={b_el_dc: solph.Flow(nominal_value=0)},
            )
        elif self.pv["settings"]["design"] == True:
            # DESIGN
            pv = solph.components.Source(
                label="pv",
                outputs={
                    b_el_dc: solph.Flow(
                        fix=self.solar_potential / self.solar_potential_peak,
                        nominal_value=None,
                        investment=solph.Investment(
                            ep_costs=self.epc["pv"] * self.n_days / 365,
                            maximum=self.demand.max() * 20
                        ),
                        variable_costs=0,
                    )
                },
            )
        else:
            # DISPATCH
            pv = solph.components.Source(
                label="pv",
                outputs={
                    b_el_dc: solph.Flow(
                        fix=self.solar_potential / self.solar_potential_peak,
                        nominal_value=self.pv["parameters"]["nominal_capacity"],
                        variable_costs=0,
                    )
                },
            )

        # -------------------- DIESEL GENSET --------------------
        # fuel density is assumed 0.846 kg/l
        fuel_cost = (
            self.diesel_genset["parameters"]["fuel_cost"]
            / 0.846
            / self.diesel_genset["parameters"]["fuel_lhv"]
        )
        fuel_source = solph.components.Source(
            label="fuel_source", outputs={b_fuel: solph.Flow(variable_costs=fuel_cost)}
        )

        # optimize capacity of the fuel generator
        self.epc["diesel_genset"] = (
            self.crf
            * self.capex_multi_investment(
                capex_0=self.diesel_genset["parameters"]["capex"],
                component_lifetime=self.diesel_genset["parameters"]["lifetime"],
            )
            + self.diesel_genset["parameters"]["opex"]
        )

        if self.diesel_genset["settings"]["is_selected"] == False:
            diesel_genset = solph.components.Transformer(
                label="diesel_genset",
                inputs={b_fuel: solph.Flow()},
                outputs={b_el_ac: solph.Flow(nominal_value=0)},
            )
        elif self.diesel_genset["settings"]["design"] == True:
            # DESIGN
            if self.diesel_genset["settings"]["offset"] is True:
                diesel_genset = solph.components.Transformer(
                    label='diesel_genset',
                    inputs={b_fuel: solph.flows.Flow()},
                    outputs={
                        b_el_ac: solph.flows.Flow(
                            nominal_value=None,
                            variable_costs=self.diesel_genset["parameters"]["variable_cost"],
                            min=self.diesel_genset["parameters"]["min_load"],
                            max=1,
                            nonconvex=solph.NonConvex(),
                            investment=solph.Investment(ep_costs=self.epc["diesel_genset"] * self.n_days / 365,
                                                        maximum=self.demand.max() * 1.2),
                        )
                    },
                    conversion_factors={b_el_ac: self.diesel_genset["parameters"]["max_efficiency"]})
            else:
                diesel_genset = solph.components.Transformer(
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
            diesel_genset = solph.components.Transformer(
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

        # -------------------- RECTIFIER --------------------
        self.epc["rectifier"] = (
            self.crf
            * self.capex_multi_investment(
                capex_0=self.rectifier["parameters"]["capex"],
                component_lifetime=self.rectifier["parameters"]["lifetime"],
            )
            + self.rectifier["parameters"]["opex"]
        )

        if self.rectifier["settings"]["is_selected"] == False:
            rectifier = solph.components.Transformer(
                label="rectifier",
                inputs={b_el_ac: solph.Flow(nominal_value=0)},
                outputs={b_el_dc: solph.Flow()},
            )
        elif self.rectifier["settings"]["design"] == True:
            # DESIGN
            rectifier = solph.components.Transformer(
                label="rectifier",
                inputs={
                    b_el_ac: solph.Flow(
                        nominal_value=None,
                        investment=solph.Investment(
                            ep_costs=self.epc["rectifier"] * self.n_days / 365,
                        maximum=self.demand.max() * 1.2
                        ),
                        variable_costs=0,
                    )
                },
                outputs={b_el_dc: solph.Flow()},
                conversion_factors={
                    b_el_dc: self.rectifier["parameters"]["efficiency"],
                },
            )
        else:
            # DISPATCH
            rectifier = solph.components.Transformer(
                label="rectifier",
                inputs={
                    b_el_ac: solph.Flow(
                        nominal_value=self.rectifier["parameters"]["nominal_capacity"],
                        variable_costs=0,
                    )
                },
                outputs={b_el_dc: solph.Flow()},
                conversion_factors={
                    b_el_dc: self.rectifier["parameters"]["efficiency"],
                },
            )

        # -------------------- INVERTER --------------------
        self.epc["inverter"] = (
            self.crf
            * self.capex_multi_investment(
                capex_0=self.inverter["parameters"]["capex"],
                component_lifetime=self.inverter["parameters"]["lifetime"],
            )
            + self.inverter["parameters"]["opex"]
        )

        if self.inverter["settings"]["is_selected"] == False:
            inverter = solph.components.Transformer(
                label="inverter",
                inputs={b_el_dc: solph.Flow(nominal_value=0)},
                outputs={b_el_ac: solph.Flow()},
            )
        elif self.inverter["settings"]["design"] == True:
            # DESIGN
            inverter = solph.components.Transformer(
                label="inverter",
                inputs={
                    b_el_dc: solph.Flow(
                        nominal_value=None,
                        investment=solph.Investment(
                            ep_costs=self.epc["inverter"] * self.n_days / 365,
                            maximum=self.demand.max() * 1.2
                        ),
                        variable_costs=0,
                    )
                },
                outputs={b_el_ac: solph.Flow()},
                conversion_factors={
                    b_el_ac: self.inverter["parameters"]["efficiency"],
                },
            )
        else:
            # DISPATCH
            inverter = solph.components.Transformer(
                label="inverter",
                inputs={
                    b_el_dc: solph.Flow(
                        nominal_value=self.inverter["parameters"]["nominal_capacity"],
                        variable_costs=0,
                    )
                },
                outputs={b_el_ac: solph.Flow()},
                conversion_factors={
                    b_el_ac: self.inverter["parameters"]["efficiency"],
                },
            )

        # -------------------- BATTERY --------------------
        self.epc["battery"] = (
            self.crf
            * self.capex_multi_investment(
                capex_0=self.battery["parameters"]["capex"],
                component_lifetime=self.battery["parameters"]["lifetime"],
            )
            + self.battery["parameters"]["opex"]
        )

        if self.battery["settings"]["is_selected"] == False:
            battery = solph.components.GenericStorage(
                label="battery",
                nominal_storage_capacity=0,
                inputs={b_el_dc: solph.Flow()},
                outputs={b_el_dc: solph.Flow()},
            )
        elif self.battery["settings"]["design"] == True:
            # DESIGN
            battery = solph.components.GenericStorage(
                label="battery",
                nominal_storage_capacity=None,
                investment=solph.Investment(
                    ep_costs=self.epc["battery"] * self.n_days / 365,
                    maximum=self.demand.max() * 48
                ),
                inputs={b_el_dc: solph.Flow(variable_costs=0)},
                outputs={b_el_dc: solph.Flow(investment=solph.Investment(ep_costs=0))},
                initial_storage_level=self.battery["parameters"]["soc_min"],
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
            battery = solph.components.GenericStorage(
                label="battery",
                nominal_storage_capacity=self.battery["parameters"]["nominal_capacity"],
                inputs={b_el_dc: solph.Flow(variable_costs=0)},
                outputs={b_el_dc: solph.Flow()},
                initial_storage_level=self.battery["parameters"]["soc_min"],
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

        # -------------------- DEMAND --------------------
        demand_el = solph.components.Sink(
            label="electricity_demand",
            inputs={
                b_el_ac: solph.Flow(
                    # min=1-max_shortage_timestep,
                    fix=self.demand / self.demand_peak,
                    nominal_value=self.demand_peak,
                )
            },
        )

        # -------------------- SURPLUS --------------------
        surplus = solph.components.Sink(
            label="surplus",
            inputs={b_el_ac: solph.Flow()},
        )

        # -------------------- SHORTAGE --------------------
        # maximal unserved demand and the variable costs of unserved demand.
        if self.shortage["settings"]["is_selected"]:
            shortage = solph.components.Source(
                label="shortage",
                outputs={
                    b_el_ac: solph.Flow(
                        variable_costs=self.shortage["parameters"][
                            "shortage_penalty_cost"
                        ],
                        nominal_value=self.shortage["parameters"]["max_shortage_total"]
                        * sum(self.demand),
                        full_load_time_max=1,
                    ),
                },
            )
        else:
            shortage = solph.components.Source(
                label="shortage",
                outputs={
                    b_el_ac: solph.Flow(
                        nominal_value=0,
                    ),
                },
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
            surplus,
            shortage,
        )

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

        # def max_surplus_electricity_total_rule(model):
        #     max_surplus_electricity = 0.05  # fraction
        #     expr = 0
        #     ## ------- Get generated at t ------- #
        #     generated_diesel_genset = sum(model.flow[diesel_genset, b_el_ac, :])
        #     generated_pv = sum(model.flow[inverter, b_el_ac, :])
        #     ac_to_dc = sum(model.flow[b_el_ac, rectifier, :])
        #     generated = generated_diesel_genset + generated_pv - ac_to_dc
        #     expr += (generated * max_surplus_electricity)
        #     ## ------- Get surplus at t------- #
        #     surplus_total = sum(model.flow[b_el_ac, surplus, :])
        #     expr += -surplus_total

        #     return expr >= 0

        # model.max_surplus_electricity_total = po.Constraint(
        #     rule=max_surplus_electricity_total_rule
        # )

        # optimize the energy system
        # gurobi --> 'MipGap': '0.01'
        # cbc --> 'ratioGap': '0.01'
        solver_option = {"gurobi": {"MipGap": "0.03"}, "cbc": {"ratioGap": "0.03"}}

        model.solve(solver=self.solver,
            solve_kwargs={"tee": True},
            cmdline_options=solver_option[self.solver],)
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
        results_surplus = solph.views.node(results=self.results_main, node="surplus")
        results_shortage = solph.views.node(results=self.results_main, node="shortage")

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

        # hourly profiles for surplus ac and dc electricity production
        self.sequences_surplus = results_surplus["sequences"][
            (("electricity_ac", "surplus"), "flow")
        ]

        # hourly profiles for shortages in the demand coverage
        self.sequences_shortage = results_shortage["sequences"][
            (("shortage", "electricity_ac"), "flow")
        ]

        # -------------------- SCALARS (STATIC) --------------------
        if self.diesel_genset["settings"]["is_selected"] == False:
            self.capacity_genset = 0
        elif self.diesel_genset["settings"]["design"] == True:
            self.capacity_genset = results_diesel_genset["scalars"][
                (("diesel_genset", "electricity_ac"), "invest")
            ]
        else:
            self.capacity_genset = self.diesel_genset["parameters"]["nominal_capacity"]

        if self.pv["settings"]["is_selected"] == False:
            self.capacity_pv = 0
        elif self.pv["settings"]["design"] == True:
            self.capacity_pv = results_pv["scalars"][
                (("pv", "electricity_dc"), "invest")
            ]
        else:
            self.capacity_pv = self.pv["parameters"]["nominal_capacity"]

        if self.inverter["settings"]["is_selected"] == False:
            self.capacity_inverter = 0
        elif self.inverter["settings"]["design"] == True:
            self.capacity_inverter = results_inverter["scalars"][
                (("electricity_dc", "inverter"), "invest")
            ]
        else:
            self.capacity_inverter = self.inverter["parameters"]["nominal_capacity"]

        if self.rectifier["settings"]["is_selected"] == False:
            self.capacity_rectifier = 0
        elif self.rectifier["settings"]["design"] == True:
            self.capacity_rectifier = results_rectifier["scalars"][
                (("electricity_ac", "rectifier"), "invest")
            ]
        else:
            self.capacity_rectifier = self.rectifier["parameters"]["nominal_capacity"]

        if self.battery["settings"]["is_selected"] == False:
            self.capacity_battery = 0
        elif self.battery["settings"]["design"] == True:
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

        self.surplus_rate = (
            100
            * self.sequences_surplus.sum(axis=0)
            / (
                self.sequences_genset.sum(axis=0)
                - self.sequences_rectifier.sum(axis=0)
                + self.sequences_inverter.sum(axis=0)
            )
        )
        self.genset_to_dc = (
            100
            * self.sequences_rectifier.sum(axis=0)
            / self.sequences_genset.sum(axis=0)
        )
        self.shortage = (
            100
            * self.sequences_shortage.sum(axis=0)
            / self.sequences_demand.sum(axis=0)
        )

        print("")
        print(40 * "*")
        print(f"LCOE:\t\t {self.lcoe:.2f} cent/kWh")
        print(f"RES:\t\t {self.res:.0f}%")
        print(f"Surplus:\t {self.surplus_rate:.1f}% of the total production")
        print(f"Shortage:\t {self.shortage:.1f}% of the total demand")
        print(f"AC--DC:\t\t {self.genset_to_dc:.1f}% of the genset production")
        print(40 * "*")
        print(f"genset:\t\t {self.capacity_genset:.0f} kW")
        print(f"pv:\t\t {self.capacity_pv:.0f} kW")
        print(f"st:\t\t {self.capacity_battery:.0f} kW")
        print(f"inv:\t\t {self.capacity_inverter:.0f} kW")
        print(f"rect:\t\t {self.capacity_rectifier:.0f} kW")
        print(f"peak:\t\t {self.sequences_demand.max():.0f} kW")
        print(f"surplus:\t {self.sequences_surplus.max():.0f} kW")
        print(40 * "*")
