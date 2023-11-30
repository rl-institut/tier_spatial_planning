import copy
import math
import time
import numpy as np
import pandas as pd
from pyproj import Proj
from k_means_constrained import KMeansConstrained
from scipy.sparse.csgraph import minimum_spanning_tree
from fastapi_app.helper.error_logger import logger as error_logger
from fastapi_app.db import sync_inserts, sa_tables, sync_queries
from fastapi_app.opt_models.base_optimizer import BaseOptimizer

pd.options.mode.chained_assignment = None  # default='warn'


def optimize_grid(user_id, project_id):
    try:
        grid_opt = GridOptimizer(user_id, project_id)
        grid_opt.optimize_grid()
        grid_opt.results_to_db()
    except Exception as exc:
        user_name = 'user with user_id: {}'.format(user_id)
        error_logger.error_log(exc, 'no request', user_name)
        raise exc


class GridOptimizer(BaseOptimizer):

    def __init__(self, user_id, project_id):
        super().__init__(user_id,
                         project_id, )
        self._update_project_status_in_db()
        self.user_id = user_id
        self.project_id = project_id
        self.nodes, self.power_house = self._query_nodes()
        self.links = pd.DataFrame(
            {"label": pd.Series([], dtype=str),
             "lat_from": pd.Series([], dtype=np.dtype(float)),
             "lon_from": pd.Series([], dtype=np.dtype(float)),
             "lat_to": pd.Series([], dtype=np.dtype(float)),
             "lon_to": pd.Series([], dtype=np.dtype(float)),
             "x_from": pd.Series([], dtype=np.dtype(float)),
             "y_from": pd.Series([], dtype=np.dtype(float)),
             "x_to": pd.Series([], dtype=np.dtype(float)),
             "y_to": pd.Series([], dtype=np.dtype(float)),
             "link_type": pd.Series([], dtype=str),
             "length": pd.Series([], dtype=int),
             "n_consumers": pd.Series([], dtype=int),
             "total_power": pd.Series([], dtype=int),
             "from_node": pd.Series([], dtype=str),
             "to_node": pd.Series([], dtype=str), }).set_index("label")
        self.pole_max_connection = self.project_setup["pole_max_n_connections"]
        self.grid_mst = pd.DataFrame({}, dtype=np.dtype(float))
        self.epc_distribution_cable = self.calc_epc(self.project_setup["distribution_cable_capex"],
                                                    self.project_setup["distribution_cable_lifetime"])
        self.epc_connection_cable = self.calc_epc(self.project_setup["connection_cable_capex"],
                                                  self.project_setup["connection_cable_lifetime"])
        self.epc_connection = self.calc_epc(self.project_setup["mg_connection_cost"],
                                            self.project_setup["project_lifetime"])
        self.epc_pole = self.calc_epc(self.project_setup["pole_capex"],
                                      self.project_setup["pole_lifetime"])
        self.max_levelized_grid_cost = self.project_setup["shs_max_grid_cost"]
        self.connection_cable_max_length = self.project_setup["connection_cable_max_length"]
        self.distribution_cable_max_length = self.project_setup["distribution_cable_max_length"]

    def optimize_grid(self):
        self.convert_lonlat_xy()
        self._clear_poles()
        n_total_consumers = self.nodes.index.__len__()
        n_shs_consumers = self.nodes[self.nodes["is_connected"] == False].index.__len__()
        n_grid_consumers = n_total_consumers - n_shs_consumers
        self.nodes.sort_index(key=lambda x: x.astype("int64"), inplace=True)
        if self.power_house is not None:
            power_house_consumers = self._connect_power_house_consumer_manually(self.connection_cable_max_length)
            self._placeholder_consumers_for_power_house()
        else:
            power_house_consumers = None
        n_poles = self._find_opt_number_of_poles(n_grid_consumers)
        self.determine_poles(min_n_clusters=n_poles,
                             power_house_consumers=power_house_consumers)
        # Find the connection links_df in the network with lengths greater than the
        # maximum allowed length for `connection` cables, specified by the user.
        long_links = self.find_index_longest_distribution_link()
        # Add poles to the identified long `distribution` links_df, so that the
        # distance between all poles remains below the maximum allowed distance.
        self._add_fixed_poles_on_long_links(long_links=long_links)
        # Update the (lon,lat) coordinates based on the newly inserted poles
        # which only have (x,y) coordinates.
        self.convert_lonlat_xy(inverse=True)
        # Connect all poles together using the minimum spanning tree algorithm.
        self.connect_grid_poles(long_links=long_links)
        # Calculate distances of all poles from the load centroid.
        # Find the location of the power house.
        self.add_number_of_distribution_and_connection_cables()
        n_iter = 2 if self.power_house is None else 1
        for i in range(n_iter):
            if self.power_house is None and i == 0:
                self._select_location_of_power_house()
            self._set_direction_of_links()
            self.allocate_poles_to_branches()
            self.allocate_subbranches_to_branches()
            self.label_branch_of_consumers()
            self.determine_cost_per_pole()
            self._connection_cost_per_consumer()
            self.determine_costs_per_branch()
            consumer_idxs = self.nodes[self.nodes['node_type'] == 'consumer'].index
            self.nodes.loc[consumer_idxs, 'yearly_consumption'] = self.demand.sum() / len(consumer_idxs)
            self._determine_shs_consumers()
            if self.power_house is None and self.links.index.__len__() > 0:
                old_power_house = self.nodes[self.nodes["node_type"] == 'power-house'].index[0]
                self._select_location_of_power_house()
                new_power_house = self.nodes[self.nodes["node_type"] == 'power-house'].index[0]
                if old_power_house == new_power_house:
                    break
            else:
                break

    def results_to_db(self):
        self._nodes_to_db()
        self._links_to_db()
        self._results_to_db()

    def _nodes_to_db(self):
        nodes_df = self.nodes.reset_index(drop=True)
        nodes_df.drop(labels=["x", "y", "cluster_label", "type_fixed", "n_connection_links", "n_distribution_links",
                              "cost_per_pole", "branch", "parent_branch", "total_grid_cost_per_consumer_per_a",
                              "connection_cost_per_consumer", 'cost_per_branch', 'distribution_cost_per_branch',
                              'yearly_consumption'],
                      axis=1,
                      inplace=True)
        nodes_df = nodes_df.round(decimals=6)
        if not nodes_df.empty:
            nodes_df.latitude = nodes_df.latitude.map(lambda x: "%.6f" % x)
            nodes_df.longitude = nodes_df.longitude.map(lambda x: "%.6f" % x)
            if len(nodes_df.index) != 0:
                if 'parent' in nodes_df.columns:
                    nodes_df['parent'] = nodes_df['parent'].where(nodes_df['parent'] != 'unknown', None)
                nodes = sa_tables.Nodes()
                nodes.id = self.user_id
                nodes.project_id = self.project_id
                nodes.data = nodes_df.reset_index(drop=True).to_json()
                sync_inserts.merge_model(nodes)

    def _links_to_db(self):
        links_df = self.links.reset_index(drop=True)
        links_df.drop(labels=["x_from", "y_from", "x_to", "y_to", "n_consumers", "total_power", "from_node", "to_node"],
                      axis=1,
                      inplace=True)
        links_df.lat_from = links_df.lat_from.map(lambda x: "%.6f" % x)
        links_df.lon_from = links_df.lon_from.map(lambda x: "%.6f" % x)
        links_df.lat_to = links_df.lat_to.map(lambda x: "%.6f" % x)
        links_df.lon_to = links_df.lon_to.map(lambda x: "%.6f" % x)
        links = sa_tables.Links()
        links.id = self.user_id
        links.project_id = self.project_id
        links.data = links_df.reset_index(drop=True).to_json()
        sync_inserts.merge_model(links)

    def _results_to_db(self):
        results = sa_tables.Results()
        results.n_consumers = len(self.consumers())
        results.n_shs_consumers = self.nodes[self.nodes["is_connected"] == False].index.__len__()
        results.n_poles = len(self._poles())
        results.length_distribution_cable = int(self.links[self.links.link_type == "distribution"]["length"].sum())
        results.length_connection_cable = int(self.links[self.links.link_type == "connection"]["length"].sum())
        results.cost_grid = int(self.cost()) if self.links.index.__len__() > 0 else 0
        results.cost_shs = 0
        results.time_grid_design = round(time.monotonic() - self.start_execution_time, 1)
        results.n_distribution_links = int(self.links[self.links["link_type"] == "distribution"].shape[0])
        results.n_connection_links = int(self.links[self.links["link_type"] == "connection"].shape[0])
        results.id = self.user_id
        results.project_id = self.project_id
        sync_inserts.merge_model(results)

    def _update_project_status_in_db(self):
        self.start_execution_time = time.monotonic()
        print('start grid opt')
        project_setup = sync_queries.get_project_setup_of_user(self.user_id, self.project_id)
        project_setup.status = "in progress"
        sync_inserts.merge_model(project_setup)

    def _query_nodes(self):
        nodes_df = self.nodes.copy()
        nodes_df.loc[nodes_df['shs_options'] == 2, 'is_connected'] = False
        nodes_df['is_connected'] = True
        nodes_df.index = nodes_df.index.astype(str)
        nodes_df = nodes_df[nodes_df['node_type'].isin(['consumer', 'power-house'])]
        power_house = nodes_df.loc[nodes_df['node_type'] == 'power-house']
        if power_house.index.__len__() > 0 and power_house['how_added'].iat[0] != 'manual':
            nodes_df = nodes_df.drop(index=power_house.index)
            power_house = None
        elif power_house.index.__len__() == 0:
            power_house = None
        return nodes_df, power_house

    # -------------------- NODES -------------------- #
    def _get_load_centroid(self):
        """
        This function obtains the ideal location for the power house, which is
        at the load centroid of the village.
        """
        grid_consumers = self.nodes[self.nodes["is_connected"] == True]
        lat = np.average(grid_consumers["latitude"])
        lon = np.average(grid_consumers["longitude"])
        self.load_centroid = [lat, lon]

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        radius = 6371.0 * 1000

        # Convert degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Differences
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        # Haversine formula
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Distance
        distance = radius * c
        return distance

    def _get_poles_distances_from_load_centroid(self):
        """
        This function calculates all distances between the poles and the load
        centroid of the settlement.
        """
        self._get_load_centroid()
        lat2, lon2 = self.load_centroid
        for pole_index in self._poles().index:
            lat1 = self.nodes.latitude.loc[pole_index]
            lon1 = self.nodes.longitude.loc[pole_index]
            self.nodes.distance_to_load_center.loc[pole_index] = GridOptimizer.haversine_distance(lat1, lon1, lat2,
                                                                                                  lon2)

    def _select_location_of_power_house(self):
        """
        This function assumes the closest pole to the calculated location for
        the power house, as the new location of the power house.
        """
        self._get_poles_distances_from_load_centroid()
        poles_with_consumers = self._poles()
        poles_with_consumers = poles_with_consumers[poles_with_consumers["n_connection_links"] > 0]
        min_distance_nearest_pole = poles_with_consumers["distance_to_load_center"].min()
        nearest_pole = self._poles()[self._poles()["distance_to_load_center"] == min_distance_nearest_pole]
        self.nodes.loc[self.nodes["node_type"] == 'power-house', 'node_type'] = "pole"
        self.nodes.loc[nearest_pole.index, "node_type"] = "power-house"

    def _connect_power_house_consumer_manually(self, max_length):
        power_house = self.nodes.loc[self.nodes['node_type'] == 'power-house']
        self.convert_lonlat_xy()
        x2 = power_house['x'].values[0]
        y2 = power_house['y'].values[0]
        for consumer in self.nodes[self.nodes["node_type"] == 'consumer'].index:
            x1 = self.nodes.x.loc[consumer]
            y1 = self.nodes.y.loc[consumer]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            self.nodes.loc[consumer, "distance_to_load_center"] = distance
        consumers = self.nodes[self.nodes["distance_to_load_center"] <= max_length].copy()
        if consumers.index.__len__() > 0:
            self.nodes = self.nodes.drop(index=consumers.index)
        return consumers

    def _placeholder_consumers_for_power_house(self, remove=False):
        n_max = self.pole_max_connection
        label_start = 100000
        power_house = self.nodes.loc[self.nodes['node_type'] == 'power-house']
        for i in range(n_max):
            label = str(label_start + i)
            if remove is True:
                self.nodes = self.nodes.drop(index=label)
            else:
                self._add_node(label,
                               latitude=power_house['latitude'].values[0],
                               longitude=power_house['longitude'].values[0],
                               x=np.nan,
                               y=np.nan,
                               cluster_label=np.nan,
                               n_connection_links=np.nan,
                               n_distribution_links=np.nan,
                               parent=np.nan)
        if remove is False:
            self.convert_lonlat_xy()

    def _clear_nodes(self):
        """
        Removes all nodes from the grid.
        """
        self.nodes = self.nodes.drop([label for label in self.nodes.index], axis=0)

    def _clear_poles(self):
        """
        Removes all poles from the grid.
        """
        self.nodes = self.nodes.drop(
            [
                label
                for label in self.nodes.index
                if self.nodes.node_type.loc[label] in ["pole"]
            ],
            axis=0,
        )

    def get_grid_consumers(self):
        df = self.nodes[(self.nodes["is_connected"] == True) & (self.nodes["node_type"] == 'consumer')]
        return df.copy()

    def get_shs_consumers(self):
        df = self.nodes[(self.nodes["is_connected"] == False) & (self.nodes["node_type"] == 'consumer')]
        return df.copy()

    def find_index_longest_distribution_link(self):
        # First select the distribution links from the entire links.
        distribution_links = self.links[self.links["link_type"] == "distribution"]

        # Find the links longer than two times of the allowed distance
        critical_link = distribution_links[
            distribution_links["length"] > self.distribution_cable_max_length
            ]

        return list(critical_link.index)

    def _add_fixed_poles_on_long_links(
            self,
            long_links,
    ):

        for long_link in long_links:
            # Get start and end coordinates of the long link.
            x_from = self.links.x_from[long_link]
            x_to = self.links.x_to[long_link]
            y_from = self.links.y_from[long_link]
            y_to = self.links.y_to[long_link]

            # Calculate the number of additional poles required.
            n_required_poles = int(
                np.ceil(self.links.length[long_link] / self.distribution_cable_max_length) - 1
            )

            # Get the index of the last pole in the grid. The new pole's index
            # will start from this index.
            last_pole = self._poles().index[-1]
            # Split the pole's index using `-` as the separator, because poles
            # are labeled in `p-x` format. x represents the index number, which
            # must be an integer.
            index_last_pole = int(last_pole.split("-")[1])

            # Calculate the slope of the line, connecting the start and end
            # points of the long link.
            slope = (y_to - y_from) / (x_to - x_from)

            # Calculate the final length of the smaller links after splitting
            # the long links into smaller parts.
            length_smaller_links = self.links.length[long_link] / (n_required_poles + 1)

            # Add all poles between the start and end points of the long link.
            for i in range(1, n_required_poles + 1):
                x = x_from + np.sign(x_to - x_from) * i * length_smaller_links * np.sqrt(1 / (1 + slope ** 2))
                y = y_from + np.sign(y_to - y_from) * i * length_smaller_links * abs(slope) * np.sqrt(
                    1 / (1 + slope ** 2))

                pole_label = f"p-{i + index_last_pole}"

                # In adding the pole, the `how_added` attribute is considered
                # `long-distance-init`, which means the pole is added because
                # of long distance in a distribution link.
                # The reason for using the `long_link` part is to distinguish
                # it with the poles which are already `connected` to the grid.
                # The poles in this stage are only placed on the line, and will
                # be connected to the other poles using another function.
                # The `cluster_label` is given as 1000, to avoid inclusion in
                # other clusters.
                self._add_node(
                    label=pole_label,
                    x=x,
                    y=y,
                    node_type="pole",
                    consumer_type="n.a.",
                    consumer_detail="n.a.",
                    is_connected=True,
                    how_added=long_link,
                    type_fixed=True,
                    cluster_label=1000,
                    custom_specification='',
                    shs_options=0
                )

    def _add_node(
            self,
            label,
            latitude=0,
            longitude=0,
            x=0,
            y=0,
            node_type="consumer",
            consumer_type="household",
            consumer_detail="default",
            distance_to_load_center=0,
            is_connected=True,
            how_added="automatic",
            type_fixed=False,
            cluster_label=0,
            n_connection_links="0",
            n_distribution_links=0,
            parent="unknown",
            distribution_cost=0,
            custom_specification='',
            shs_options=0
    ):
        """
        adds a node to the grid's node dataframe.

        Parameters
        ----------
        already defined in the 'Grid' object definition
        """

        self.nodes.at[label, "longitude"] = longitude
        self.nodes.at[label, "latitude"] = latitude
        self.nodes.at[label, "x"] = x
        self.nodes.at[label, "y"] = y
        self.nodes.at[label, "node_type"] = node_type
        self.nodes.at[label, "consumer_type"] = consumer_type
        self.nodes.at[label, "consumer_detail"] = consumer_detail
        self.nodes.at[label, "distance_to_load_center"] = distance_to_load_center
        self.nodes.at[label, "is_connected"] = is_connected
        self.nodes.at[label, "how_added"] = how_added
        self.nodes.at[label, "type_fixed"] = type_fixed
        self.nodes.at[label, "cluster_label"] = cluster_label
        self.nodes.at[label, "n_connection_links"] = n_connection_links
        self.nodes.at[label, "n_distribution_links"] = n_distribution_links
        self.nodes.at[label, "parent"] = parent
        self.nodes.at[label, "distribution_cost"] = distribution_cost
        self.nodes.at[label, "custom_specification"] = custom_specification
        self.nodes.at[label, "shs_options"] = shs_options

    def consumers(self):
        """
        Returns only the 'consumer' nodes from the grid.

        Returns
        ------
        class:`pandas.core.frame.DataFrame`
            filtered pandas dataframe containing all 'consumer' nodes
        """
        return self.nodes[self.nodes["node_type"] == "consumer"]

    def _poles(self):
        """
        Returns all poles and the power house in the grid.

        Returns
        ------
        class:`pandas.core.frame.DataFrame`
            filtered pandas dataframe containing all 'pole' nodes
        """
        return self.nodes[
            (self.nodes["node_type"] == "pole")
            | (self.nodes["node_type"] == "power-house")
            ]

    def distance_between_nodes(self, label_node_1: str, label_node_2: str):
        """
        Returns the distance between two nodes of the grid

        Parameters
        ----------
        label_node_1: str
            label of the first node
        label_node_2: str
            label of the second node

        Return
        -------
            distance between the two nodes in meter
        """
        if (label_node_1 and label_node_2) in self.nodes.index:
            # (x,y) coordinates of the points
            x1 = self.nodes.x.loc[label_node_1]
            y1 = self.nodes.y.loc[label_node_1]

            x2 = self.nodes.x.loc[label_node_2]
            y2 = self.nodes.y.loc[label_node_2]

            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        else:
            return np.infty

    # -------------------- LINKS -------------------- #

    def get_links(self):
        """
        Returns all links of the grid.

        Returns
        -------
        class:`pandas.core.frame.DataFrame`
            a pandas dataframe containing all links of the grid
        """
        return self.links

    def get_nodes(self):
        """
        Return all nodes of the grid.

        Returns
        -------
        class:`pandas.core.frame.DataFrame`
            a pandas dataframe containing all nodes of the grid
        """
        return self.nodes

    def _clear_all_links(self):
        """
        Removes all links from the grid.
        """
        self.links = self.get_links().drop(
            [label for label in self.get_links().index], axis=0
        )

    def _clear_links(self, link_type):
        """
        Removes all link types given by the user from the grid.
        """
        self.links = self.get_links().drop(
            [
                label
                for label in self.get_links()[
                self.get_links()["link_type"] == link_type
                ].index
            ],
            axis=0,
        )

    def _add_links(self, label_node_from: str, label_node_to: str):
        """
        +++ ok +++

        Adds a link between two nodes of the grid and
        calculates the distance of the link.

        Parameters
        ----------
        label_node_from: str
            label of the first node
        label_node_to: str
            label of the second node

        Notes
        -----
        The method first makes sure that the two labels belong to the grid.
        Otherwise, no link will be added.
        """

        # specify the type of the link which is obtained based on the start/end nodes of the link
        if (self.nodes.node_type.loc[label_node_from]
            and self.nodes.node_type.loc[label_node_to]) == "pole":
            # convention: if two poles are getting connected, the beginning will be the one with lower number
            (label_node_from, label_node_to) = sorted([label_node_from, label_node_to])
            link_type = "distribution"
        else:
            link_type = "connection"

        # calculate the length of the link
        length = self.distance_between_nodes(
            label_node_1=label_node_from, label_node_2=label_node_to
        )

        # define a label for the link and add all other characteristics to the grid object
        label = f"({label_node_from}, {label_node_to})"
        self.links.at[label, "lat_from"] = self.nodes.latitude.loc[label_node_from]
        self.links.at[label, "lon_from"] = self.nodes.longitude.loc[label_node_from]
        self.links.at[label, "lat_to"] = self.nodes.latitude.loc[label_node_to]
        self.links.at[label, "lon_to"] = self.nodes.longitude.loc[label_node_to]
        self.links.at[label, "x_from"] = self.nodes.x.loc[label_node_from]
        self.links.at[label, "y_from"] = self.nodes.y.loc[label_node_from]
        self.links.at[label, "x_to"] = self.nodes.x.loc[label_node_to]
        self.links.at[label, "y_to"] = self.nodes.y.loc[label_node_to]
        self.links.at[label, "link_type"] = link_type
        self.links.at[label, "length"] = length
        self.links.at[label, "n_consumers"] = 0
        self.links.at[label, "total_power"] = 0
        self.links.at[label, "from_node"] = ""
        self.links.at[label, "to_node"] = ""

    def total_length_distribution_cable(self):
        """
        Calculates the total length of all cables connecting only poles in the grid.

        Returns
        ------
        type: float
            the total length of the distribution cable in the grid
        """
        return self.links.length[self.links.link_type == "distribution"].sum()

    def total_length_connection_cable(self):
        """
        Calculates the total length of all cables between each pole and
        consumers.

        Returns
        ------
        type: float
            total length of the connection cable in the grid.
        """
        return self.links.length[self.links.link_type == "connection"].sum()

    # -------------------- OPERATIONS -------------------- #

    def convert_lonlat_xy(self, inverse: bool = False):
        """
        +++ ok +++

        Converts (longitude, latitude) coordinates into (x, y)
        plane coordinates using a python package 'pyproj'.

        Parameter
        ---------
        inverse: bool (default=false)
            this parameter indicates the direction of conversion
            + false: lon,lat --> x,y
            + true: x,y --> lon/lat
        """

        p = Proj(proj="utm", zone=32, ellps="WGS84", preserve_units=False)

        # if inverse=true, this is the case when the (x,y) coordinates of the obtained
        # poles (from the optimization) are converted into (lon,lat)
        if inverse:
            # First the possible candidates for inverse conversion are picked.
            nodes_for_inverse_conversion = self.nodes[
                (self.nodes["node_type"] == "pole")
                | (self.nodes["node_type"] == "power-house")
                ]

            for node_index in nodes_for_inverse_conversion.index:
                lon, lat = p(
                    self.nodes.x.loc[node_index],
                    self.nodes.y.loc[node_index],
                    inverse=inverse,
                )
                self.nodes.at[node_index, "longitude"] = lon
                self.nodes.at[node_index, "latitude"] = lat
        else:
            for node_index in self.nodes.index:
                x, y = p(
                    self.nodes.longitude.loc[node_index],
                    self.nodes.latitude.loc[node_index],
                    inverse=inverse,
                )
                self.nodes.at[node_index, "x"] = x
                self.nodes.at[node_index, "y"] = y

            # store reference values for (x,y) to use later when converting (x,y) to (lon,lat)

    # -------------------- COSTS ------------------------ #

    def cost(self):
        """
        Computes the cost of the grid taking into account the number
        of nodes, their types (consumer or poles) and the length of
        different types of cables between nodes.

        Return
        ------
        cost of the grid
        """

        # get the number of poles, consumers and links from the grid
        n_poles = self._poles().shape[0]
        n_mg_consumers = self.consumers()[self.consumers()["is_connected"] == True].shape[0]
        n_links = self.get_links().shape[0]

        # if there is no poles in the grid, or there is no link,
        # the function returns an infinite value
        if (n_poles == 0) or (n_links == 0):
            return np.infty

        # calculate the total length of the cable used between poles [m]
        total_length_distribution_cable = self.total_length_distribution_cable()

        # calculate the total length of the `connection` cable between poles and consumers
        total_length_connection_cable = self.total_length_connection_cable()

        grid_cost = (
                n_poles * self.epc_pole
                + n_mg_consumers * self.epc_connection
                + total_length_connection_cable * self.epc_connection_cable
                + total_length_distribution_cable * self.epc_distribution_cable
        )

        return np.around(grid_cost, decimals=2)

    def add_number_of_distribution_and_connection_cables(self):
        poles = self._poles().copy()
        links = self.get_links().copy()
        links['from_node'] = pd.Series(links.index.str.split(','), index=links.index) \
            .str[0].str.replace('(', '', regex=True).str.replace(' ', '', regex=True)
        links['to_node'] = pd.Series(links.index.str.split(','), index=links.index) \
            .str[1].str.replace(')', '', regex=True).str.replace(' ', '', regex=True)
        distribution_links = links[links["link_type"] == "distribution"].copy()
        connection_links = links[links["link_type"] == "connection"].copy()
        for pole_idx in poles.index:
            n_distribution = len(distribution_links[distribution_links["from_node"] == pole_idx].index)
            n_distribution += len(distribution_links[distribution_links["to_node"] == pole_idx].index)
            self.nodes.loc[pole_idx, "n_distribution_links"] = n_distribution
            n_connection = len(connection_links[connection_links["from_node"] == pole_idx].index)
            n_connection += len(connection_links[connection_links["to_node"] == pole_idx].index)
            self.nodes.loc[pole_idx, "n_connection_links"] = n_connection
        self.nodes.loc[self.nodes["node_type"] == 'consumer', 'n_connection_links'] = 1
        self.nodes['n_distribution_links'] = self.nodes['n_distribution_links'].fillna(0).astype(int)
        self.nodes['n_connection_links'] = self.nodes['n_connection_links'].fillna(0).astype(int)

    def allocate_poles_to_branches(self):
        poles = self._poles().copy()
        links = self.get_links().copy()
        distribution_links = links[links["link_type"] == "distribution"].copy()
        leaf_poles = pd.Series(poles[poles["n_distribution_links"] == 1].index).values
        split_poles = pd.Series(poles[poles["n_distribution_links"] > 2].index).values
        power_house = poles[poles["node_type"] == "power-house"].index[0]
        start_poles = distribution_links[(distribution_links["to_node"] == power_house)]['from_node'].values
        start_set = set(start_poles)
        split_set = set(split_poles)
        diff_set = start_set - split_set
        start_poles = np.array(list(diff_set))
        for split_pole in split_poles:
            for start_pole in distribution_links[(distribution_links["to_node"] == split_pole)]['from_node'].to_list():
                start_poles = np.append(start_poles, start_pole)
        start_poles = pd.Series(start_poles).drop_duplicates().values
        self.nodes['branch'] = None
        tmp_idxs = self.nodes[self.nodes.index.isin(start_poles)].index
        self.nodes.loc[start_poles, 'branch'] = pd.Series(tmp_idxs, index=tmp_idxs)
        try:
            for start_pole in start_poles:
                next_pole = copy.deepcopy(start_pole)
                for _ in range(len(poles.index)):
                    next_pole = distribution_links[(distribution_links["to_node"] == next_pole)]['from_node']
                    if len(next_pole.index) == 1:
                        next_pole = next_pole.values[0]
                        self.nodes.loc[next_pole, 'branch'] = start_pole
                        if next_pole in split_poles or next_pole in leaf_poles:
                            break
                    else:
                        break
        except Exception as e:
            user_name = 'unknown'
            error_logger.error_log(e, 'no request', user_name)
        self.nodes.loc[
            (self.nodes['branch'].isna()) & (self.nodes['node_type'].isin(['pole', 'power-house'])), 'branch'] \
            = power_house

    def label_branch_of_consumers(self):
        branch_map = self.nodes.loc[self.nodes.node_type.isin(['pole', 'power-house']), 'branch']
        self.nodes.loc[self.nodes.node_type == 'consumer', 'branch'] \
            = self.nodes.loc[self.nodes.node_type == 'consumer', 'parent'].map(branch_map)

    def allocate_subbranches_to_branches(self):
        poles = self._poles().copy()
        self.nodes['parent_branch'] = None
        power_house = poles[poles["node_type"] == "power-house"].index[0]

        def determine_parent_branches(start_poles):
            for pole in start_poles:
                branch_start_pole = poles[poles.index == pole]['branch'].iat[0]
                split_pole = self.nodes[self.nodes.index == branch_start_pole]['parent'].iat[0]
                parent_branch = poles[poles.index == split_pole]['branch'].iat[0]
                self.nodes.loc[self.nodes['branch'] == branch_start_pole, 'parent_branch'] = parent_branch

        try:
            if len(poles['branch'].unique()) > 1:
                leaf_poles = poles[poles["n_distribution_links"] == 1].index
                determine_parent_branches(leaf_poles)
                poles_expect_power_house = poles[poles["node_type"] != "power-house"]
                split_poles = poles_expect_power_house[poles_expect_power_house["n_distribution_links"] > 2]
                if len(split_poles.index) > 0:
                    determine_parent_branches(split_poles)
        except Exception as e:
            user_name = 'unknown'
            error_logger.error_log(e, 'no request', user_name)
        self.nodes.loc[(self.nodes['parent_branch'].isna()) & (self.nodes['node_type'].isin(['pole', 'power-house'])),
        'parent_branch'] = power_house

    def determine_cost_per_pole(self):
        poles = self._poles().copy()
        links = self.get_links().copy()
        self.nodes['cost_per_pole'] = None
        self.nodes['cost_per_pole'] = self.nodes['cost_per_pole'].astype(float)
        power_house = poles[poles["node_type"] == "power-house"].index[0]
        for pole in poles.index:
            if pole != power_house:
                parent_pole = poles[poles.index == pole]['parent'].iat[0]
                try:
                    length = links[(links['from_node'] == pole) & (links['to_node'] == parent_pole)]['length'].iat[0]
                except IndexError:
                    try:
                        length = links[(links['from_node'] == parent_pole) & (links['to_node'] == pole)]['length'].iat[
                            0]
                    except Exception:
                        length = 20
                self.nodes.loc[pole, 'cost_per_pole'] = self.epc_pole + length * self.epc_distribution_cable
            else:
                self.nodes.loc[pole, 'cost_per_pole'] = self.epc_pole

    def determine_costs_per_branch(self, branch=None):
        poles = self._poles().copy()

        def _(branch):
            branch_df = self.nodes[(self.nodes['branch'] == branch) & (self.nodes['is_connected'] == True)].copy()
            cost_per_branch = self.nodes[self.nodes.index.isin(branch_df.index)]['cost_per_pole'].sum()
            cost_per_branch += self.nodes[self.nodes.index.isin(branch_df.index)]['connection_cost_per_consumer'].sum()
            self.nodes.loc[branch_df.index, 'distribution_cost_per_branch'] = cost_per_branch
            self.nodes.loc[branch_df.index, 'cost_per_branch'] = cost_per_branch

        if branch is None:
            for branch in poles['branch'].unique():
                _(branch)
        else:
            _(branch)

    def _connection_cost_per_consumer(self):
        links = self.get_links()
        grid_consumers = self.nodes[(self.nodes['node_type'] == 'consumer') &
                                    (self.nodes['is_connected'] == True)].index
        for consumer in grid_consumers:
            parent_pole = self.nodes[self.nodes.index == consumer]['parent'].iat[0]
            length = min(links[(links['from_node'] == consumer) & (links['to_node'] == parent_pole)]['length'].iat[0],
                         3)
            connection_cost = self.epc_connection + length * self.epc_connection_cable
            self.nodes.loc[consumer, 'connection_cost_per_consumer'] = connection_cost

    def get_subbranches(self, branch):
        subbranches = self.nodes[self.nodes['branch'] == branch].index.tolist()
        leaf_branches = self.nodes[self.nodes['n_distribution_links'] == 1]['branch'].index
        next_sub_branches = self.nodes[self.nodes['parent_branch'] == branch]['parent_branch'].tolist()
        for _ in range(len(self.nodes['branch'].unique())):
            next_next_sub_branches = []
            for sub_branch in next_sub_branches:
                if sub_branch in leaf_branches:
                    break
                else:
                    for b in next_sub_branches:
                        subbranches.append(b)
                    next_next_sub_branches.append(sub_branch)
            next_sub_branches = next_next_sub_branches
            if len(next_sub_branches) == 0:
                break
        return subbranches

    def get_all_consumers_of_subbranches(self, branch):
        branches = self.get_subbranches(branch)
        consumers = self.nodes[(self.nodes['node_type'] == 'consumer') &
                               (self.nodes['branch'].isin(branches)) &
                               (self.nodes['is_connected'] == True)].index
        return consumers

    def get_all_consumers_of_branch(self, branch):
        consumers = self.nodes[(self.nodes['node_type'] == 'consumer') &
                               (self.nodes['branch'].isin(branch)) &
                               (self.nodes['is_connected'] == True)].index
        return consumers

    def _determine_distribution_links(self):
        idxs = self.links[self.links.index.to_series().str.count('p') == 2].index
        self.links.loc[idxs, 'link_type'] = 'distribution'
        return idxs

    def _distribute_cost_among_consumers(self):
        self.nodes['total_grid_cost_per_consumer_per_a'] = np.nan
        self.nodes.loc[self.nodes[self.nodes['is_connected'] == True].index, 'total_grid_cost_per_consumer_per_a'] \
            = self.nodes['connection_cost_per_consumer']
        leaf_branches = self.nodes[self.nodes['n_distribution_links'] == 1]['branch'].unique()
        for branch in self.nodes['branch'].unique():
            if branch is not None:
                if branch in leaf_branches:
                    poles_of_branch = self.nodes[self.nodes['branch'] == branch]
                    next_pole = poles_of_branch[poles_of_branch['n_distribution_links'] == 1]
                    consumers_down_the_line = []
                    for _ in range(len(poles_of_branch)):
                        consumers_of_pole = poles_of_branch[(poles_of_branch['node_type'] == 'consumer') &
                                                            (poles_of_branch['is_connected'] == True) &
                                                            (poles_of_branch['parent'] == next_pole.index[0])]
                        for consumer in consumers_of_pole.index:
                            consumers_down_the_line.append(consumer)
                        total_consumption = \
                            self.nodes[self.nodes.index.isin(consumers_down_the_line)]['yearly_consumption'].sum()
                        cost_of_pole = \
                            self.nodes.loc[
                                self.nodes[self.nodes.index == next_pole.index[0]].index, 'cost_per_pole'].iat[0]
                        for consumer in consumers_down_the_line:
                            self.nodes.loc[consumer, 'total_grid_cost_per_consumer_per_a'] += \
                                cost_of_pole * self.nodes.loc[consumer, 'yearly_consumption'] / total_consumption
                        next_pole = self.nodes[self.nodes.index == next_pole['parent'].iat[0]]
                        if next_pole.index.__len__() == 0:
                            break
                        elif self.nodes[self.nodes.index == next_pole.index[0]]['branch'].iat[0] != branch:
                            break
                else:
                    continue
        self.nodes['total_grid_cost_per_consumer_per_a'] = \
            self.nodes['total_grid_cost_per_consumer_per_a'] / self.nodes['yearly_consumption']

    def marginal_cost_per_consumer(self, pole, consumer_of_pole):
        total_consumption = \
            self.nodes[self.nodes.index.isin(consumer_of_pole.index)]['yearly_consumption'].sum()
        cost_of_pole = \
            self.nodes.loc[self.nodes[self.nodes.index == pole].index, 'cost_per_pole'].iat[0]
        connection_cost_consumers \
            = self.nodes[self.nodes.index.isin(consumer_of_pole.index)]['connection_cost_per_consumer'].sum()
        next_pole = self.nodes[self.nodes.index == pole]['parent'].iat[0]
        for _ in range(100):
            if next_pole == 'unknown':
                continue
            if self.nodes[self.nodes.index == next_pole]['n_connection_links'].iat[0] == 0:
                if self.nodes[self.nodes.index == next_pole]['node_type'].iat[0] == 'power-house':
                    break
                cost_of_pole += \
                    self.nodes.loc[self.nodes[self.nodes.index == next_pole].index, 'cost_per_pole'].iat[0]
                next_pole = self.nodes[self.nodes.index == next_pole]['parent'].iat[0]
            else:
                break
        marginal_cost_of_pole = (cost_of_pole + connection_cost_consumers) / (total_consumption + 0.0000001)
        return marginal_cost_of_pole

    def _cut_leaf_poles_on_condition(self):
        exclude_lst = [self.nodes[self.nodes['node_type'] == 'power-house'].index[0]]
        for pole in self.nodes[self.nodes['shs_options'] == 1]['parent'].unique():
            exclude_lst.append(pole)
        for _ in range(100):
            counter = 0
            leaf_poles = self.nodes[self.nodes['n_distribution_links'] == 1].index
            for pole in leaf_poles:
                if pole in exclude_lst:
                    continue
                consumer_of_pole = self.nodes[self.nodes['parent'] == pole]
                branch = self.nodes[self.nodes.index == pole]['branch'].iat[0]
                consumer_of_branch = self.nodes[self.nodes['branch'] == branch].index
                average_total_cost_of_pole = consumer_of_pole['total_grid_cost_per_consumer_per_a'].mean()
                average_marginal_cost_of_pole = self.marginal_cost_per_consumer(pole, consumer_of_pole)
                self.determine_costs_per_branch(branch)
                average_marginal_branch_cost_of_pole = self.nodes.loc[consumer_of_branch, 'cost_per_branch'].iat[0] \
                                                       / (self.nodes.loc[
                                                              consumer_of_branch, 'yearly_consumption'].sum() + 1e-9)
                if average_marginal_cost_of_pole > self.max_levelized_grid_cost:
                    self._cut_specific_pole(pole)
                    counter += 1
                elif average_total_cost_of_pole > self.max_levelized_grid_cost and \
                        average_marginal_branch_cost_of_pole > self.max_levelized_grid_cost:
                    self._cut_specific_pole(pole)
                    counter += 1
                else:
                    exclude_lst.append(pole)
            if counter == 0:
                break

    def _cut_specific_pole(self, pole):
        mask = (self.nodes['parent'] == pole) & \
               (self.nodes['node_type'] == 'consumer')
        self.nodes.loc[mask, 'is_connected'] = False
        self.nodes.loc[mask, 'parent'] = np.nan
        self.nodes.loc[mask, 'branch'] = np.nan
        self.nodes.loc[mask, 'total_grid_cost_per_consumer_per_a'] = np.nan
        self._remove_poles_and_links(pole)
        self._cut_leaf_poles_without_connection()

    def _cut_leaf_poles_without_connection(self):
        exclude_lst = [self.nodes[self.nodes['node_type'] == 'power-house'].index[0]]
        for _ in range(len(self.nodes[self.nodes['node_type'] == 'pole'])):
            leaf_poles = self.nodes[self.nodes['n_distribution_links'] == 1].index
            counter = 0
            for pole in leaf_poles:
                if pole in exclude_lst:
                    continue
                consumer_of_pole = self.nodes[self.nodes['parent'] == pole]
                if len(consumer_of_pole.index) == 0:
                    self._remove_poles_and_links(pole)
                    counter += 1
                else:
                    exclude_lst.append(pole)
            if counter == 0:
                break

    def _remove_poles_and_links(self, pole):
        self._correct_n_distribution_links_of_parent_poles(pole)
        self.nodes = self.nodes.drop(index=pole)
        drop_idxs = self.links[(self.links['from_node'] == pole) | (self.links['to_node'] == pole)].index
        self.links = self.links.drop(index=drop_idxs)

    def _correct_n_distribution_links_of_parent_poles(self, pole):
        parent_pole = self.nodes[self.nodes.index == pole]['parent'].iat[0]
        self.nodes.loc[parent_pole, 'n_distribution_links'] -= 1

    def _determine_shs_consumers(self, max_iter=20):
        for _ in range(max_iter):
            self._distribute_cost_among_consumers()
            if self.nodes['total_grid_cost_per_consumer_per_a'].max() < self.max_levelized_grid_cost:
                if self.nodes['n_connection_links'].sum() == 0:
                    self.nodes = \
                        self.nodes.drop(index=self.nodes[self.nodes['node_type'].isin(['power-house', 'pole'])].index)
                    self.links = self.links.drop(index=self.links.index)
                break
            self._cut_leaf_poles_on_condition()
        self._remove_power_house_if_no_poles_connected()

    def _remove_power_house_if_no_poles_connected(self):
        if self.nodes[self.nodes['node_type'] == 'pole'].empty:
            self.nodes = self.nodes[self.nodes['node_type'] == 'consumer']
            self.links = self.links.drop(index=self.links.index)
            self.nodes['is_connected'] = False

    def _set_direction_of_links(self):
        consumer_to_power_house = True  # if True, direction is from consumer to power-house
        self._determine_distribution_links()
        links = self.get_links().copy()
        links["poles"] = links.index.str.replace('[\(\) ]', '', regex=True)
        distribution_links = links[links["link_type"] == "distribution"].copy()
        poles = self._poles().copy()
        power_house_idx = poles[poles["node_type"] == "power-house"].index[0]
        parent_pole_list = [power_house_idx]
        examined_pole_list = []

        def change_direction_of_links(from_pole, to_pole, links):
            row_idxs = '({}, {})'.format(from_pole, to_pole)
            if row_idxs not in links.index:
                row_idxs = '({}, {})'.format(to_pole, from_pole)
            if from_pole in row_idxs.split(',')[1]:
                new_row_idxs = '({}, {})'.format(from_pole, to_pole)
                links = links.rename(index={row_idxs: new_row_idxs})
            return links

        def check_all_child_poles(parent_pole_list, links, examined_pole_list):
            new_parent_pole_list = []
            for parent_pole in parent_pole_list:
                child_pole_list \
                    = distribution_links[(distribution_links['poles'].str.contains(parent_pole + ',')) |
                                         ((distribution_links['poles'] + '#').str.contains(parent_pole + '#'))] \
                    ['poles'].str.split(',')
                for child_pole in child_pole_list:
                    if child_pole[0] not in examined_pole_list and child_pole[1] not in examined_pole_list:
                        pos = 0 if consumer_to_power_house else 1
                        if child_pole[pos] == parent_pole:
                            tmp_parent_pole = copy.deepcopy(child_pole[0])
                            child_pole = child_pole[1]
                        else:
                            tmp_parent_pole = copy.deepcopy(child_pole[1])
                            child_pole = child_pole[0]
                        self.nodes.loc[child_pole, 'parent'] = tmp_parent_pole
                        new_parent_pole_list.append(child_pole)
                        links = change_direction_of_links(child_pole, parent_pole, links)
            examined_pole_list += parent_pole_list
            return new_parent_pole_list, links, examined_pole_list

        def check_all_parent_poles(child_pole_list, links, examined_pole_list):
            new_child_pole_list = []
            for child_pole in child_pole_list:
                parent_pole_list = distribution_links[(distribution_links['poles'].str.contains(child_pole + ',')) | \
                                                      ((distribution_links['poles'] + '#').str.contains(
                                                          child_pole + '#'))] \
                    ['poles'].str.split(',')
                for parent_pole in parent_pole_list:
                    if parent_pole[0] not in examined_pole_list and parent_pole[1] not in examined_pole_list:
                        pos = 0 if consumer_to_power_house else 1
                        if parent_pole[pos] == child_pole:
                            tmp_child_pole = copy.deepcopy(parent_pole[0])
                            parent_pole = parent_pole[1]
                        else:
                            tmp_child_pole = copy.deepcopy(parent_pole[1])
                            parent_pole = parent_pole[0]
                        self.nodes.loc[tmp_child_pole, 'parent'] = parent_pole
                        new_child_pole_list.append(parent_pole)
                        links = change_direction_of_links(child_pole, parent_pole, links)
            examined_pole_list += child_pole_list
            return new_child_pole_list, links, examined_pole_list

        for _ in range(len(links.index)):
            if len(parent_pole_list) > 0:
                parent_pole_list, links, examined_pole_list \
                    = check_all_child_poles(parent_pole_list, links, examined_pole_list)
            else:
                self.nodes.loc[self.nodes['node_type'] == 'power-house', 'parent'] = \
                    self.nodes[self.nodes['node_type'] == 'power-house'].index[0]
                if self.nodes['parent'][self.nodes['parent'] == 'unknown'].__len__() == 0:
                    break
                else:
                    child_pole_list = self.nodes[(self.nodes['parent'] == 'unknown') &
                                                 (self.nodes['n_distribution_links'] == 1) &
                                                 (self.nodes['node_type'] == 'pole')].index.tolist()
                    for __ in range(len(links.index)):
                        if len(child_pole_list) > 0 and \
                                self.nodes['parent'][self.nodes['parent'] == 'unknown'].__len__() > 0:
                            child_pole_list, links, examined_pole_list \
                                = check_all_parent_poles(child_pole_list, links, examined_pole_list)
                        else:
                            break

        links['from_node'] = pd.Series(links.index.str.split(','), index=links.index) \
            .str[0].str.replace('(', '', regex=True).str.replace(' ', '', regex=True)
        links['to_node'] = pd.Series(links.index.str.split(','), index=links.index) \
            .str[1].str.replace(')', '', regex=True).str.replace(' ', '', regex=True)
        links = links.drop(columns=["poles"])
        if consumer_to_power_house:
            mask = links['link_type'] == 'connection'
            links.loc[mask, ['from_node', 'to_node']] = links.loc[mask, ['to_node', 'from_node']].values
        self.links = links.copy(True)

    def calc_epc(self, capex_0, component_lifetime):
        epc = (self.crf * BaseOptimizer.capex_multi_investment(self,
                                                               capex_0=capex_0,
                                                               component_lifetime=component_lifetime)) * self.n_days / 365
        return epc

    # ------------ CONNECT NODES USING TREE-STAR SHAPE ------------#
    def connect_grid_consumers(self):
        """
        +++ ok +++

        This method create the connections between each consumer and the
        nearest pole


        Parameters
        ----------
        grid (~grids.Grid):
            grid object
        """
        # Remove all existing connections between poles and consumers
        self._clear_links(link_type="connection")

        # calculate the number of clusters and their labels obtained from kmeans clustering
        n_clusters = self._poles()[self._poles()["type_fixed"] == False].shape[0]
        cluster_labels = self._poles()["cluster_label"]

        # create links between each node and the corresponding centroid
        for cluster in range(n_clusters):

            if self.nodes[self.nodes["cluster_label"] == cluster].index.__len__() == 1:
                continue

            # first filter the nodes and only select those with cluster labels equal to 'cluster'
            filtered_nodes = self.nodes[self.nodes["cluster_label"] == cluster_labels[cluster]]

            # then obtain the label of the pole which is in this cluster (as the center)
            pole_label = filtered_nodes.index[filtered_nodes["node_type"] == "pole"][0]

            for node_label in filtered_nodes.index:
                # adding consumers
                if node_label != pole_label:
                    if self.nodes.loc[node_label, "is_connected"]:
                        self._add_links(label_node_from=str(pole_label), label_node_to=str(node_label))
                        self.nodes.loc[node_label, "parent"] = str(pole_label)

    def connect_grid_poles(self, long_links=None):
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
        long_links = list() if long_links is None else long_links
        # First, all links in the grid should be removed.
        self._clear_links(link_type="distribution")

        # Now, all links from the sparse matrix obtained using the minimum
        # spanning tree are stored in `links_mst`.
        # All poles in the `links_mst` should be connected together considering:
        #   + The number of rows in the 'links_mst' reveals the number of
        #     connections.
        #   + (x,y) of each nonzero element of the 'links_mst' correspond to the
        #     (pole_from, pole_to) labels.
        links_mst = np.argwhere(self.grid_mst != 0)

        for link_mst in range(links_mst.shape[0]):
            mst_pole_from = self._poles().index[links_mst[link_mst, 0]]
            mst_pole_to = self._poles().index[links_mst[link_mst, 1]]

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
                added_poles_from_to = self._poles()[
                    (self._poles()["type_fixed"] == True)
                    & (self._poles()["how_added"] == mst_from_to)
                    ]
                added_poles_to_from = self._poles()[
                    (self._poles()["type_fixed"] == True)
                    & (self._poles()["how_added"] == mst_to_from)
                    ]

                # In addition to the `added_poles` a flag is defined here to
                # deal with the direction of adding additional poles.
                if not added_poles_from_to.empty:
                    added_poles = added_poles_from_to
                    to_from = False
                elif not added_poles_to_from.empty:
                    added_poles = added_poles_to_from
                    to_from = True
                else:
                    raise UnboundLocalError('\'added_poles\' unkown')

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
                            self._add_links(
                                label_node_from=index_added_pole,
                                label_node_to=mst_pole_to,
                            )
                        else:
                            self._add_links(
                                label_node_from=mst_pole_from,
                                label_node_to=index_added_pole,
                            )

                    if counter == n_added_poles - 1:
                        # The last `added poles` should be connected to
                        # the end or to the beginning of the long link,
                        # depending on the `to_from` flag.
                        if to_from:
                            self._add_links(
                                label_node_from=mst_pole_from,
                                label_node_to=index_added_pole,
                            )
                        else:
                            self._add_links(
                                label_node_from=index_added_pole,
                                label_node_to=mst_pole_to,
                            )

                    if counter > 0:
                        # The intermediate `added poles` should connect to
                        # the other `added_poles` before and after them.
                        self._add_links(
                            label_node_from=added_poles.index[counter - 1],
                            label_node_to=added_poles.index[counter],
                        )
                    counter += 1

                    # Change the `how_added` tag for the new poles.
                    self.nodes.at[index_added_pole, "how_added"] = "long-distance"

            # If `link_mst` does not belong to the list of long links, it is
            # simply connected without any further check.
            else:
                self._add_links(
                    label_node_from=self._poles().index[links_mst[link_mst, 0]],
                    label_node_to=self._poles().index[links_mst[link_mst, 1]],
                )

    def create_minimum_spanning_tree(self):
        """
        Creates links between all poles using the Kruskal's algorithm for
        the minimum spanning tree method from scipy.sparse.csgraph.

        Parameters
        ----------
        grid (~grids.Grid):
            grid object
        """

        # total number of poles (i.e., clusters)
        poles = self._poles()
        n_poles = poles.shape[0]

        # generate all possible edges between each pair of poles
        graph_matrix = np.zeros((n_poles, n_poles))
        for i in range(n_poles):
            for j in range(n_poles):
                # since the graph does not have a direction, only the upper part of the matrix must be filled
                if j > i:
                    graph_matrix[i, j] \
                        = self.distance_between_nodes(label_node_1=poles.index[i], label_node_2=poles.index[j])
        # obtain the optimal links between all poles (grid_mst) and copy it in the grid object
        grid_mst = minimum_spanning_tree(graph_matrix)
        self.grid_mst = grid_mst

    #  --------------------- K-MEANS CLUSTERING ---------------------#
    def kmeans_clustering(self, n_clusters: int):
        """
        Uses a k-means clustering algorithm and returns the coordinates of the centroids.

        Parameters
        ----------
            grid (~grids.Grid):
                grid object
            n_cluster (int):
                number of clusters (i.e., k-value) for the k-means clustering algorithm

        Return
        ------
            coord_centroids: numpy.ndarray
                A numpy array containing the coordinates of the cluster centroids.
                Suppose there are two cluster with centers at (x1, y1) & (x2, y2),
                then the output array would look like:
                    array([
                        [x1, y1],
                        [x2 , y2]
                        ])
        """

        # first, all poles must be removed from the nodes list
        self._clear_poles()
        grid_consumers = self.get_grid_consumers()

        # gets (x,y) coordinates of all nodes in the grid
        nodes_coord = np.array([[grid_consumers.x.loc[index], grid_consumers.y.loc[index]]
                                for index in grid_consumers.index if grid_consumers.is_connected.loc[index] == True])

        # call kmeans clustering with constraints (min and max number of members in each cluster )
        kmeans = KMeansConstrained(
            n_clusters=n_clusters,
            init="k-means++",  # 'k-means++' or 'random'
            n_init=10,
            max_iter=300,
            tol=1e-4,
            size_min=0,
            size_max=self.pole_max_connection,
            random_state=0,
            n_jobs=5,
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
        self.nodes = pd.concat([grid_consumers, poles, self.get_shs_consumers()], axis=0)

        # compute (lon,lat) coordinates for the poles
        self.convert_lonlat_xy(inverse=True)

    def determine_poles(self, min_n_clusters, power_house_consumers):
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
            the number of poles corresponding to the minimum cost of the grid
        """
        # obtain the location of poles using kmeans clustering method
        self.kmeans_clustering(n_clusters=min_n_clusters)
        # create the minimum spanning tree to obtain the optimal links between poles
        if self.power_house is not None:
            cluster_label = self.nodes.loc['100000', 'cluster_label']
            power_house_idx = self.nodes[(self.nodes["node_type"] == "pole") &
                                         (self.nodes["cluster_label"] == cluster_label)].index
            power_house_consumers['cluster_label'] = cluster_label
            power_house_consumers['consumer_type'] = np.nan
            self.nodes = pd.concat([self.nodes, power_house_consumers], )
            self._placeholder_consumers_for_power_house(remove=True)

        self.create_minimum_spanning_tree()

        # connect all links in the grid based on the previous calculations
        self.connect_grid_consumers()
        self.connect_grid_poles()
        if self.power_house is not None:
            self.nodes.loc[self.nodes.index == power_house_idx[0], "node_type"] = 'power-house'
            self.nodes.loc[self.nodes.index == power_house_idx[0], "how_added"] = 'manual'

    def _find_opt_number_of_poles(self, n_mg_consumers):
        # calculate the minimum number of poles based on the
        # maximum number of connections at each pole
        if self.pole_max_connection == 0:
            min_number_of_poles = 1
        else:
            min_number_of_poles = int(np.ceil(n_mg_consumers / self.pole_max_connection))

        space = pd.Series(range(min_number_of_poles, n_mg_consumers, 1))

        def is_enough_poles(n):
            self.kmeans_clustering(n_clusters=n)
            self.connect_grid_consumers()
            constraints_violation = self.links[self.links["link_type"] == "connection"]
            constraints_violation = constraints_violation[
                constraints_violation["length"] > self.connection_cable_max_length]
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
