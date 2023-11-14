import copy
import math
import numpy as np
import pandas as pd
from configparser import ConfigParser
import os
from pyproj import Proj
from fastapi_app.tools.error_logger import logger as error_logger
pd.options.mode.chained_assignment = None  # default='warn'


def make_folder(folder):
    """
    If no folder of the given name already exists, create new one.

    Parameters
    ----------
    folder: str
        Name of the folder to be created.
    """

    if not os.path.exists(folder):
        parent_folders = folder.split('/')
        for i in range(1, len(parent_folders) + 1):
            path = ''
            for x in parent_folders[0:i]:
                path += x + '/'
            if not os.path.exists(path[0:-1]):
                os.mkdir(path[0:-1])

class Grid:
    """
    Defines a basic grid containing all the information about the topology
    of the network. The grid contains a network representation composed of
    a set of nodes and a set of links.

    Attributes
    ----------
    grid_id: str
        name of the grid

    nodes: :class:`pandas.core.frame.DataFrame`
        pandas dataframe containing the following information
        related to the nodes (consumers and/or poles):
            - label
            - coordinates
                + latitude & longitude
                + x & y
            - node_type ('consumer' or 'pole')
            - consumer_type ('household', 'enterprise', ...)
            - consumer_detail ('default' or 'custom')
            - is_connected (is False only for solar-home-system nodes)
            - how_added (if a node is added as a result of optimization or by user)
            - segment (which cluster the node belongs to)
            - allocation capacity, denoting how many consumers can be
              connected to the node.

    links : :class:`pandas.core.frame.DataFrame`
        Table containing the following information related to the links:
            - label
            - lat_from
            - lat_to
            - long_from
            - long_to
            - link_type
                + 'connection': between poles and consumers
                + 'distribution': between two poles
            - length

    capex: :class:`pandas.core.frame.DataFrame`
        capital expenditure associated with:
            + pole [$/pcs]
            + connection [$/pcs]
            + connection_cable [$/m]
            + distribution_cable [$/m]

    opex: :class:`pandas.core.frame.DataFrame`
        operational expenditure associated with:
            + pole [$/pcs]
            + connection [$/pcs]
            + connection_cable [$/m]
            + distribution_cable [$/m]

    pole_max_connection: int
        maximum number of consumers that can be connected to each pole.

    max_current: float
        the maximal current expected to flow into the cables [A].

    voltage: float
        the nominal (minimum) voltage that must be delivered to each consumer [V].

    cables: :class:`pandas.core.frame.DataFrame`
        a panda dataframe including the following characteristics of
        the 'distribution' and 'connection' cables:
            + cross-section area in [mm²]
            + electrical resistivity in [Ohm*mm²/m].
    """

    # -------------------- CONSTRUCTOR --------------------#
    def __init__(
        self,
        grid_id="unnamed_grid",
        nodes=pd.DataFrame(
            {
                "label": pd.Series([], dtype=str),
                "latitude": pd.Series([], dtype=np.dtype(float)),
                "longitude": pd.Series([], dtype=np.dtype(float)),
                "x": pd.Series([], dtype=np.dtype(float)),
                "y": pd.Series([], dtype=np.dtype(float)),
                "node_type": pd.Series([], dtype=str),
                "consumer_type": pd.Series([], dtype=str),
                "consumer_detail": pd.Series([], dtype=str),
                "distance_to_load_center": pd.Series([], dtype=np.dtype(float)),
                "is_connected": pd.Series([], dtype=bool),
                "how_added": pd.Series([], dtype=str),
                "type_fixed": pd.Series([], dtype=bool),
                "cluster_label": pd.Series([], dtype=np.dtype(int)),
                "n_connection_links": pd.Series([], dtype=np.dtype(str)),
                "n_distribution_links": pd.Series([], dtype=np.dtype(int)),
                "parent": pd.Series([], dtype=np.dtype(str)),
                "distribution_cost": pd.Series([], dtype=np.dtype(float)),
            }
        ).set_index("label"),
        ref_node=np.zeros(2),
        links=pd.DataFrame(
            {
                "label": pd.Series([], dtype=str),
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
                "to_node": pd.Series([], dtype=str),
            }
        ).set_index("label"),
        epc_distribution_cable=2,  # per meter
        epc_connection_cable=0.5,  # per meter
        epc_connection=12,
        epc_pole=100,
        pole_max_connection=0,
        max_current=10,  # [A]
        voltage=230,  # [V]
        grid_mst=pd.DataFrame({}, dtype=np.dtype(float)),
        cables=pd.DataFrame(
            {
                "cable_type": pd.Series(["distribution", "connection"], dtype=str),
                "section_area": pd.Series([4, 2.5], dtype=np.dtype(float)),  # [mm²]
                "resistivity": pd.Series(
                    [0.0171, 0.0171], dtype=np.dtype(float)
                ),  # [Ohm*mm²/m]
            }
        ),
        max_levelized_grid_cost = 0.20  # [$ per consumer per year)]
    ):
        self.id = grid_id
        self.nodes = nodes
        self.ref_node = ref_node
        self.links = links
        self.pole_max_connection = pole_max_connection
        self.max_current = max_current
        self.voltage = voltage
        self.cables = cables
        self.grid_mst = grid_mst
        self.epc_distribution_cable = epc_distribution_cable  # per meter
        self.epc_connection_cable = epc_connection_cable  # per meter
        self.epc_connection = epc_connection
        self.epc_pole = epc_pole
        self.max_levelized_grid_cost = max_levelized_grid_cost

    # -------------------- NODES -------------------- #
    def get_load_centroid(self):
        """
        This function obtains the ideal location for the power house, which is
        at the load centroid of the village.
        """
        grid_consumers = self.nodes[self.nodes["is_connected"] == True]
        lat = np.average(grid_consumers["latitude"])
        lon = np.average(grid_consumers["longitude"])
        self.load_centroid = [lat, lon]

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0 * 1000

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
        distance = R * c
        return distance

    def get_poles_distances_from_load_centroid(self):
        """
        This function calculates all distances between the poles and the load
        centroid of the settlement.
        """
        self.get_load_centroid()
        lat2, lon2 = self.load_centroid
        for pole_index in self.poles().index:
            lat1 = self.nodes.latitude.loc[pole_index]
            lon1 = self.nodes.longitude.loc[pole_index]
            self.nodes.distance_to_load_center.loc[pole_index] = self.haversine_distance(lat1, lon1, lat2, lon2)

    def select_location_of_power_house(self):
        """
        This function assumes the closest pole to the calculated location for
        the power house, as the new location of the power house.

        TODO:
        If the length of the cable connecting the power house and the nearest
        pole is longer than the maximum allowed distance for distribution links,
        some more poles will be placed on it.
        """
        self.get_poles_distances_from_load_centroid()
        poles_with_consumers = self.poles()
        poles_with_consumers = poles_with_consumers[poles_with_consumers["n_connection_links"] > 0]
        min_distance_nearest_pole = poles_with_consumers["distance_to_load_center"].min()
        nearest_pole = self.poles()[self.poles()["distance_to_load_center"] == min_distance_nearest_pole]
        self.nodes.loc[self.nodes["node_type"] == 'power-house', 'node_type'] = "pole"
        self.nodes.loc[nearest_pole.index, "node_type"] = "power-house"

    def connect_power_house_consumer_manually(self, max_length):
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

    def placeholder_consumers_for_power_house(self, remove=False):
        n_max = self.pole_max_connection
        label_start = 100000
        power_house = self.nodes.loc[self.nodes['node_type'] == 'power-house']
        for i in range(n_max):
            label = str(label_start + i)
            if remove is True:
                self.nodes = self.nodes.drop(index=label)
            else:
                self.add_node(label,
                              latitude=power_house['latitude'].values[0],
                              longitude=power_house['longitude'].values[0],
                              x = np.nan,
                              y = np.nan,
                              cluster_label = np.nan,
                              n_connection_links = np.nan,
                              n_distribution_links = np.nan,
                              parent = np.nan)
        if remove is False:
            self.convert_lonlat_xy()

    def clear_nodes(self):
        """
        Removes all nodes from the grid.
        """
        self.nodes = self.nodes.drop([label for label in self.nodes.index], axis=0)

    def clear_poles(self):
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


    def find_index_longest_distribution_link(self, max_distance_dist_links):
        # First select the distribution links from the entire links.
        distribution_links = self.links[self.links["link_type"] == "distribution"]

        # Find the links longer than two times of the allowed distance
        critical_link = distribution_links[
            distribution_links["length"] > max_distance_dist_links
        ]

        return list(critical_link.index)


    def add_fixed_poles_on_long_links(
        self,
        long_links,
        max_allowed_distance,
    ):

        for long_link in long_links:
            # Get start and end coordinates of the long link.
            x_from = self.links.x_from[long_link]
            x_to = self.links.x_to[long_link]
            y_from = self.links.y_from[long_link]
            y_to = self.links.y_to[long_link]

            # Calculate the number of additional poles required.
            n_required_poles = int(
                np.ceil(self.links.length[long_link] / max_allowed_distance) - 1
            )

            # Get the index of the last pole in the grid. The new pole's index
            # will start from this index.
            last_pole = self.poles().index[-1]
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
                x = x_from + np.sign(x_to - x_from) * i * length_smaller_links * np.sqrt(1 / (1 + slope**2))
                y = y_from + np.sign(y_to - y_from) * i * length_smaller_links * abs(slope) * np.sqrt(1 / (1+ slope**2))

                pole_label = f"p-{i+index_last_pole}"

                # In adding the pole, the `how_added` attribute is considerd
                # `long-distance-init`, which means the pole is added because
                # of long distance in a distribution link.
                # The reason for using the `long_link` part is to distinguish
                # it with the poles which are already `connected` to the grid.
                # The poles in this stage are only placed on the line, and will
                # be connected to the other poles using another function.
                # The `cluster_label` is given as 1000, to avoid inclusion in
                # other clusters.
                self.add_node(
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

    def add_node(
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

    def poles(self):
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

    def clear_all_links(self):
        """
        Removes all links from the grid.
        """
        self.links = self.get_links().drop(
            [label for label in self.get_links().index], axis=0
        )

    def clear_links(self, link_type):
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

    def add_links(self, label_node_from: str, label_node_to: str):
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
            # convention: if two poles are getting connected, the begining will be the one with lower number
            (label_node_from, label_node_to) = sorted([label_node_from, label_node_to])
            link_type = "distribution"
        else:
            link_type = "connection"

        # calculate the length of the link
        length = self.distance_between_nodes(
            label_node_1=label_node_from, label_node_2=label_node_to
        )

        # define a label for the link and add all other charateristics to the grid object
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
                    self.nodes.x.loc[node_index] + self.ref_node[0],
                    self.nodes.y.loc[node_index] + self.ref_node[1],
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
            if self.ref_node[0] != 0 and self.ref_node[1]!= 0:
                self.ref_node[0] = min(self.nodes.x)
                self.ref_node[1] = min(self.nodes.y)

                # change absolute (x,y) to relative (x,y) to make them smaller and more readable
                self.nodes.x -= self.ref_node[0]
                self.nodes.y -= self.ref_node[1]

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
        n_poles = self.poles().shape[0]
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

    def get_id(self):
        """
        Returns __id attribute of the grid.

        Returns
        ------
        str
            __id parameter of the grid
        """
        return self.id

    # ----------------- COMPUTE DISTANCE BETWEEN NODES -----------------#

    def get_cable_distance_from_consumers_to_pole(self):
        """
        This method computes the cable distance separating each node
        from its powerhub. It recursively uses the method
        measure_distance_for_next_node() to explore the tree starting from
        the powerhub and following each tree branch until all nodes are
        reached.

        Returns
        ------
        class:`pandas.core.frame.DataFrame`
            This method returns a pandas DataFrame containing all the
            nodes in the grid and the total length of distribution and
            connection cable separating it from its respective powerhub.
        """

        # Create dataframe with distribution and connection cable length
        distance_df = pd.DataFrame(
            {
                "label": [],
                "distribution cable [m]": [],
                "connection cable [m]": [],
                "powerhub label": [],
            }
        )
        distance_df = distance_df.set_index("label")
        # For every powerhub, compute cable length to nodes from the segment
        for index_powerhub in self.get_nodes()[self.get_nodes()["node_type"].isin(['power-house'])].index:
            distance_df.loc[index_powerhub] = [0, 0, index_powerhub]
            # this list gathers the index of all nodes that are directly
            # connected with a link to the powerhub
            node_next_neighbours = []
            # add all nodes connected to the pole to the list
            links = self.get_links()
            for next_node in links[
                (links["from_node"] == index_powerhub)
            ]["to_node"]:
                if (
                    next_node not in node_next_neighbours
                    and next_node not in distance_df.index
                ):
                    node_next_neighbours.append(next_node)
            for next_node in links[
                (links["to_node"] == index_powerhub)
            ]["from_node"]:
                if (
                    next_node not in node_next_neighbours
                    and next_node not in distance_df.index
                ):
                    node_next_neighbours.append(next_node)
            # Call measure_distance_for_next_node for all branches
            for node in node_next_neighbours:
                self.measure_distance_for_next_node(
                    index_powerhub, node, distance_df, index_powerhub
                )
        return distance_df


    def add_number_of_distribution_and_connection_cables(self):
        poles = self.poles().copy()
        links = self.get_links().copy()
        links['from_node'] = pd.Series(links.index.str.split(','), index=links.index)\
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
        poles = self.poles().copy()
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
        self.nodes.loc[(self.nodes['branch'].isna()) & (self.nodes['node_type'].isin(['pole', 'power-house'])), 'branch'] \
            = power_house

    def label_branch_of_consumers(self):
        branch_map = self.nodes.loc[self.nodes.node_type.isin(['pole', 'power-house']), 'branch']
        self.nodes.loc[self.nodes.node_type == 'consumer', 'branch'] \
            = self.nodes.loc[self.nodes.node_type == 'consumer', 'parent'].map(branch_map)

    def allocate_subbranches_to_branches(self):
        poles = self.poles().copy()
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
        poles = self.poles().copy()
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
                        length = links[(links['from_node'] == parent_pole) & (links['to_node'] == pole)]['length'].iat[0]
                    except Exception:
                        length = 20
                self.nodes.loc[pole, 'cost_per_pole'] = self.epc_pole + length * self.epc_distribution_cable
            else:
                self.nodes.loc[pole, 'cost_per_pole'] = self.epc_pole

    def determine_costs_per_branch(self, branch=None):
        poles = self.poles().copy()

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


    def connection_cost_per_consumer(self):
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

    def determine_distribution_links(self):
        idxs = self.links[self.links.index.to_series().str.count('p') == 2].index
        self.links.loc[idxs, 'link_type'] = 'distribution'
        return idxs

    def distribute_cost_among_consumers(self):
        self.nodes['total_grid_cost_per_consumer_per_a'] = np.nan
        self.nodes.loc[self.nodes[self.nodes['is_connected'] == True].index, 'total_grid_cost_per_consumer_per_a'] \
            = self.nodes['connection_cost_per_consumer']
        leaf_branches = self.nodes[self.nodes['n_distribution_links'] == 1]['branch'].unique()
        next_branch = False
        for branch in self.nodes['branch'].unique():

            if branch is not None:
                if branch in leaf_branches:
                    poles_of_branch = self.nodes[self.nodes['branch'] == branch]
                    next_pole = poles_of_branch[poles_of_branch['n_distribution_links'] == 1]
                    consumers_down_the_line = []
                    next_branch = False
                    for _ in range(len(poles_of_branch)):
                        if next_branch:
                            break
                        consumers_of_pole = poles_of_branch[(poles_of_branch['node_type'] == 'consumer') &
                                                            (poles_of_branch['is_connected'] == True) &
                                                            (poles_of_branch['parent'] == next_pole.index[0])]
                        for consumer in consumers_of_pole.index:
                            consumers_down_the_line.append(consumer)
                        total_consumption = \
                            self.nodes[self.nodes.index.isin(consumers_down_the_line)]['yearly_consumption'].sum()
                        cost_of_pole = \
                            self.nodes.loc[self.nodes[self.nodes.index == next_pole.index[0]].index, 'cost_per_pole'].iat[0]
                        for consumer in consumers_down_the_line:
                            self.nodes.loc[consumer, 'total_grid_cost_per_consumer_per_a'] += \
                                cost_of_pole * self.nodes.loc[consumer, 'yearly_consumption'] / total_consumption
                        next_pole = self.nodes[self.nodes.index == next_pole['parent'].iat[0]]
                        if next_pole.index.__len__() == 0:
                            next_branch = True
                            break
                        elif self.nodes[self.nodes.index == next_pole.index[0]]['branch'].iat[0]  != branch:
                            break
                else:
                    continue
                    """
                    consumers = self.get_all_consumers_of_subbranches(branch)
                    total_consumption = self.nodes[self.nodes.index.isin(consumers)]['yearly_consumption'].sum()
                    cost_of_branch = \
                    self.nodes.loc[self.nodes[self.nodes['branch'] == branch].index, 'distribution_cost_per_branch'].iat[0]
                    for consumer in consumers:
                        self.nodes.loc[consumer, 'total_grid_cost_per_consumer_per_a'] += \
                            cost_of_branch * self.nodes.loc[consumer, 'yearly_consumption'] / total_consumption
                    """
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


    def cut_leaf_poles_on_condition(self):
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
                       / (self.nodes.loc[consumer_of_branch, 'yearly_consumption'].sum() + 1e-9)
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
        self.remove_poles_and_links(pole)
        self.cut_leaf_poles_without_connection()

    def cut_leaf_poles_without_connection(self):
        exclude_lst = [self.nodes[self.nodes['node_type'] == 'power-house'].index[0]]
        for _ in range(len(self.nodes[self.nodes['node_type'] == 'pole'])):
            leaf_poles = self.nodes[self.nodes['n_distribution_links'] == 1].index
            counter = 0
            for pole in leaf_poles:
                if pole in exclude_lst:
                    continue
                consumer_of_pole = self.nodes[self.nodes['parent'] == pole]
                if len(consumer_of_pole.index) == 0:
                    self.remove_poles_and_links(pole)
                    counter += 1
                else:
                    exclude_lst.append(pole)
            if counter == 0:
                break

    def remove_poles_and_links(self, pole):
        self.correct_n_distribution_links_of_parent_poles(pole)
        self.nodes = self.nodes.drop(index=pole)
        drop_idxs = self.links[(self.links['from_node'] == pole) | (self.links['to_node'] == pole)].index
        self.links = self.links.drop(index=drop_idxs)

    def correct_n_distribution_links_of_parent_poles(self, pole):
        parent_pole = self.nodes[self.nodes.index == pole]['parent'].iat[0]
        self.nodes.loc[parent_pole, 'n_distribution_links'] -= 1


    def determine_shs_consumers(self, max_iter=20):
        for _ in range(max_iter):
            self.distribute_cost_among_consumers()
            if self.nodes['total_grid_cost_per_consumer_per_a'].max() < self.max_levelized_grid_cost:
                if self.nodes['n_connection_links'].sum() == 0:
                    self.nodes = \
                        self.nodes.drop(index=self.nodes[self.nodes['node_type'].isin(['power-house', 'pole'])].index)
                    self.links = self.links.drop(index=self.links.index)
                break
            self.cut_leaf_poles_on_condition()
        self.remove_power_house_if_no_poles_connected()

    def remove_power_house_if_no_poles_connected(self):
        if self.nodes[self.nodes['node_type'] == 'pole'].empty:
            self.nodes = self.nodes[self.nodes['node_type'] == 'consumer']
            self.links = self.links.drop(index=self.links.index)
            self.nodes['is_connected'] = False


    def set_direction_of_links(self):
        consumer_to_power_house = True  # if True, direction is from consumer to power-house
        self.determine_distribution_links()
        links = self.get_links().copy()
        links["poles"] = links.index.str.replace('[\(\) ]', '', regex=True)
        distribution_links = links[links["link_type"] == "distribution"].copy()
        poles = self.poles().copy()
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
                = distribution_links[(distribution_links['poles'].str.contains(parent_pole+',')) |
                                     ((distribution_links['poles'] + '#').str.contains(parent_pole + '#'))]\
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
                                    ((distribution_links['poles'] + '#').str.contains(child_pole + '#'))] \
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

        links['from_node'] = pd.Series(links.index.str.split(','), index=links.index)\
            .str[0].str.replace('(', '', regex=True).str.replace(' ', '', regex=True)
        links['to_node'] = pd.Series(links.index.str.split(','), index=links.index)\
            .str[1].str.replace(')', '', regex=True).str.replace(' ', '', regex=True)
        links = links.drop(columns=["poles"])
        if consumer_to_power_house:
            mask = links['link_type'] == 'connection'
            links.loc[mask, ['from_node', 'to_node']] = links.loc[mask, ['to_node', 'from_node']].values
        self.links = links.copy(True)


    def measure_distance_for_next_node(self, node_n_minus_1, node_n, distance_df, index_powerhub):
        """
        This method is used to measure the cable distance between each nodes
        and the powerhub. It is designed to be recursively called to explore
        all the branches of the tree taking the powerhub as the starting point
        and exploring every branch and sub-branches until the distance to every
        node has been computed. It takes advantage that the network is a tree,
        it is thus possible to explore the branches without considering each
        node more than once.
        Parameters
        ----------
        index_powerhub
        node_n_minus_1: str
            index corresponding to the node at the base of the branch leading
            to the "node_n" (which is the node whose distance to powerhub
            has to be computed).
        node_n: str
            index corresponding to the node whose distance to powerhub has to
            be computed.
        distance_df: :class:`pandas.core.frame.DataFrame`
            dictionnary containing the distance to the powerhub of all nodes
            that where already computed using the function.
        """

        # find out what the link index of the link between nodes node_n_minus_1
        # and node_n is. Since nodes are undirected, we need to look for the
        # link from node_n_minus_1 to node_n and from node_n and node_n_minus_1
        if (
            self.get_links()[
                (self.get_links()["from_node"] == node_n_minus_1)
                & (self.get_links()["to_node"] == str(node_n))
            ].shape[0]
            == 1
        ):
            index_link_between_nodes = self.get_links()[
                (self.get_links()["from_node"] == node_n_minus_1)
                & (self.get_links()["to_node"] == str(node_n))
            ].index[0]

        elif (
            self.get_links()[
                (self.get_links()["from_node"] == node_n)
                & (self.get_links()["to_node"] == node_n_minus_1)
            ].shape[0]
            == 1
        ):
            index_link_between_nodes = self.get_links()[
                (self.get_links()["from_node"] == node_n)
                & (self.get_links()["to_node"] == node_n_minus_1)
            ].index[0]

        # check what type the link is to know distiguish of the cable types
        # in the datafram
        if self.get_links()["link_type"][index_link_between_nodes] == "distribution":
            distance_df.loc[node_n] = [
                distance_df["distribution cable [m]"][node_n_minus_1]
                + self.distance_between_nodes(node_n_minus_1, node_n),
                0,
                index_powerhub,
            ]
        elif self.get_links()["link_type"][index_link_between_nodes] == "connection":
            distance_df.loc[node_n] = [
                distance_df["distribution cable [m]"][node_n_minus_1],
                self.distance_between_nodes(node_n_minus_1, node_n),
                index_powerhub,
            ]

        # Call function for all the nodes that were not measured yet
        node_next_neighbours = []
        for node_next_neighbour in self.get_links()[
            (self.get_links()["from_node"] == node_n)
        ]["to_node"]:
            if (
                node_next_neighbour not in node_next_neighbours
                and node_next_neighbour not in distance_df.index
            ):
                node_next_neighbours.append(node_next_neighbour)
        for node_next_neighbour in self.get_links()[(self.get_links()["to_node"] == node_n)]["from_node"]:
            if (
                node_next_neighbour not in node_next_neighbours
                and node_next_neighbour not in distance_df.index
            ):
                node_next_neighbours.append(node_next_neighbour)
        for next_neighbour in node_next_neighbours:
            self.measure_distance_for_next_node(
                node_n, next_neighbour, distance_df, index_powerhub
            )

    # -------------------- GRID PERFORMANCE ---------------------- #

    def get_voltage_drop_at_nodes(self):
        """
        This method computes the voltage drop at each node using the
        parameters defined in config_grid.cfg under [power flow].

        Returns
        ------
        class:`pandas.core.frame.DataFrame`
            pandas DataFrame containing the cable distance for the different
            types of cables as well as the cable resistance between the node
            and the corresponding powerhub. The DataFrame also contains an
            estimation of the voltage drop and the voltage drop fraction.

        Notes
        -----
            The cable resistance R_i is computed as follows:
            R_i =  rho_i * 2* d_i / (i_cable_section)
            where i represent the cable type, rho the cable electric
            resistivity (self.distribution_cable_resistivity),
            d the cable distance and i_cable_section the section of the cable.
            The voltage drop is computed using Ohm's law
            U = R * I where U is the tension (here corresponding to the
            voltage drop), R the resistance and I the current.
        """
        voltage_drop_df = self.get_cable_distance_from_consumers_to_pole()

        voltage_drop_df["distribution cable resistance [Ω]"] = (
            self.cables[self.cables["cable_type"] == "distribution"]["resistivity"].iat[0]
            * 2
            * voltage_drop_df["distribution cable [m]"]
            / self.cables[self.cables["cable_type"] == "distribution"]["section_area"].iat[0]
        )

        voltage_drop_df["connection cable resistance [Ω]"] = (
            self.cables[self.cables["cable_type"] == "connection"]["resistivity"].iat[0]
            * 2
            * voltage_drop_df["connection cable [m]"]
            / self.cables[self.cables["cable_type"] == "connection"]["section_area"].iat[0]
        )

        voltage_drop_df["voltage drop [V]"] = (
            voltage_drop_df["distribution cable resistance [Ω]"] * self.max_current
        ) + (voltage_drop_df["connection cable resistance [Ω]"] * self.max_current)

        voltage_drop_df["voltage drop fraction [%]"] = (
            100 * voltage_drop_df["voltage drop [V]"] / self.voltage
        )
        if voltage_drop_df["voltage drop fraction [%]"].max() > 100:
            # ToDo: Voltage drop fraction should be between 0 and 100.
            voltage_drop_df["voltage drop fraction [%]"] = '-'
        return voltage_drop_df

    def export(
        self,
        backup_name=None,
        folder=None,
        allow_saving_in_existing_backup_folder=False,
        save_image=True,
    ):
        """
        Method calling the export_grid function to save a backup of the grid.

                Definition of the exprt_grid function:
                Export grid in folder as separated files:
                - nodes.csv
                    contains the __nodes attribute data.
                - links.csv
                    contains the __links attribute data.
                - grid_attributes.cfg
                    contains the value of the Grid's attributes.

            Parameters
            ----------
            grid: :class:`~grids.Grid`
                    Grid object.
            folder: str
                Path of the folder the grid should be saved in
            backup_name: str
                Name of the grid backup.

            allow_saving_in_existing_backup_folder: bool
                When True and a folder with the same name as the parameter
                backup_name, no new folder is created and the grid is exported in
                the folder of the backup_name.
                When False and a folder with the same name as the parameter
                backup_name exists, a new folder is created with a extension _i
                (where i is an integer).

            Notes
            -----
                If no folder is given, the default path to folder is
                f'data/backup/{grid._id}/'.

                If no folder name is given, the grid will be saved in a folder called
                f'backup_{grid._id}_{counter}', where counter is a index added
                to distiguish backups of the same grid.


        """
        export_grid(
            self,
            backup_name=backup_name,
            folder=folder,
            allow_saving_in_existing_backup_folder=(
                allow_saving_in_existing_backup_folder
            ),
        )


# - FUNCTIONS RELATED TO EXPORTING AND IMPORTING GRIDS FROM EXTERNAL FILE --#


def export_grid(
    grid, backup_name=None, folder=None, allow_saving_in_existing_backup_folder=False
):
    """
    Export grid in folder as separated files:
        - nodes.csv
            contains the __nodes attribute data.
        - links.csv
            contains the __links attribute data.
        - grid_attributes.cfg
            contains the value of the Grid's attributes.

    Parameters
    ----------
    grid: :class:`~grids.Grid`
            Grid object.
    folder: str
        Path of the folder the grid should be saved in
    backup_name: str
        Name of the grid backup.

    allow_saving_in_existing_backup_folder: bool
        When True and a folder with the same name as the parameter
        backup_name, no new folder is created and the grid is exported in
        the folder of the backup_name.
        When False and a folder with the same name as the parameter
        backup_name exists, a new folder is created with a extension _i
        (where i is an integer).

    Notes
    -----
        If no folder is given, the default path to folder is
        f'data/backup/{grid._id}/'.

        If no folder name is given, the grid will be saved in a folder called
        f'backup_{grid._id}_{counter}', where counter is a index added
        to distiguish backups of the same grid.
    """

    if folder is None:
        folder = "data/backup/" + grid.get_id()
        make_folder("data")
        make_folder("data/backup")
        make_folder(folder)
    else:
        if not os.path.exists(folder):
            parent_folders = folder.split("/")
            for i in range(1, len(parent_folders) + 1):
                path = ""
                for x in parent_folders[0:i]:
                    path += x + "/"
                make_folder(path[0:-1])

    if backup_name is None:
        backup_name = f"backup_{grid.get_id()}"

    if not allow_saving_in_existing_backup_folder:
        if os.path.exists(f"{folder}/{backup_name}"):
            counter = 1
            while os.path.exists(f"{folder}/{backup_name}_{counter}"):
                counter += 1
            backup_name = f"{backup_name}_{counter}"
    full_path = f"{folder}/{backup_name}"

    make_folder(full_path)

    # Export nodes dataframe into csv file
    grid.get_nodes().to_csv(full_path + "/nodes.csv")

    # Export links dataframe into csv file
    grid.get_links().to_csv(full_path + "/links.csv")

    # Create config files containing Grid's attributes
    config = ConfigParser()

    config["attributes"] = {
        key: (value, type(value))
        for key, value in grid.__dict__.items()
        if key not in ["_Grid__nodes", "_Grid__links"]
    }

    with open(f"{full_path}/grid_attributes.cfg", "w") as f:
        config.write(f)

    print(f"Grid saved in \n{full_path}\n\n")


def import_grid(folder):
    """
    Import a grid that was previously exported using the export_grid
    function.

    Parameters
    ----------
    folder: str
        Path to the folder where the backup files (nodes.csv, links.csv
        and attributes.cfg)
        of the given grid have been saved in.
    Returns
    -------
        Copy of the exported Grid located in the folder defined by the path.
    """

    # Import grid id from grid_attributes.cfg file
    config_attributes = ConfigParser()

    config_attributes.read(folder + "/grid_attributes.cfg")
    attributes = {}

    # Load all attributes names, values and type and fill attributes dict
    for section in config_attributes.sections():
        for (key, val) in config_attributes.items(section):
            value, str_type = (
                config_attributes.get(section, key)
                .replace("'", "")
                .replace("(", "")
                .replace(">)", "")
                .replace(" ", "")
                .replace("<class", "")
                .split(",")
            )
            if str_type == "int":
                value = int(value)
            if str_type == "float":
                value = float(value)
            # Format class name from _grid__ to _Grid__
            key_in_list = [letter for letter in key]
            key_in_list[1] = key_in_list[1].upper()
            key_formated = "".join(key_in_list)
            attributes[key_formated] = value

    # Import data from csv files to nodes and links DataFrames
    nodes = pd.read_csv(
        folder + "/nodes.csv",
        converters={
            "label": str,
            "x_coordinate": float,
            "y_coordinate": float,
            "node_type": str,
            "type_fixed": lambda x: True if x == "True" else False,
            "segment": str,
            "allocation_capacity": int,
        },
    )
    nodes = nodes.set_index("label")

    links = pd.read_csv(
        folder + "/links.csv",
        converters={
            "from": str,
            "to": str,
            "type": str,
            "distance": float,
        },
        index_col=[0],
    )
    # Create new Grid containing nodes and links
    grid = Grid(nodes=nodes, links=links)
    # Set grid attributes using dict created from grid_config.cfg
    for key, value in attributes.items():
        setattr(grid, key, value)

    return grid
