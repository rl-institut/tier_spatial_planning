from operator import inv, length_hint
from turtle import distance
import numpy as np
import pandas as pd
from fastapi_app.tools.io import make_folder
from configparser import ConfigParser
import os
from pyproj import Proj


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
            - average_consumption (in kWh per year)
            - peak_demand (in kW)
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
                "surface_area": pd.Series([], dtype=np.dtype(float)),
                "peak_demand": pd.Series([], dtype=np.dtype(float)),
                "average_consumption": pd.Series([], dtype=np.dtype(float)),
                "distance_to_load_center": pd.Series([], dtype=np.dtype(float)),
                "is_connected": pd.Series([], dtype=bool),
                "how_added": pd.Series([], dtype=str),
                "type_fixed": pd.Series([], dtype=bool),
                "cluster_label": pd.Series([], dtype=np.dtype(int)),
                "segment": pd.Series([], dtype=np.dtype(str)),
                "allocation_capacity": pd.Series([], dtype=np.dtype(int)),
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

    # -------------------- NODES -------------------- #
    def get_load_centroid(self):
        """
        This function obtains the ideal location for the power house, which is
        at the load centroid of the village.
        """
        x_centroid = np.average(self.nodes["x"], weights=self.nodes["peak_demand"])
        y_centroid = np.average(self.nodes["y"], weights=self.nodes["peak_demand"])
        self.load_centroid = [x_centroid, y_centroid]

    def get_nodes_distances_from_load_centroid(self):
        """
        This function calculates all distances between the nodes and the load
        centroid of the village.
        """
        for node_index in self.nodes.index:
            x_node = self.nodes.x.loc[node_index]
            y_node = self.nodes.y.loc[node_index]

            distance = np.sqrt(
                (x_node - self.load_centroid[0]) ** 2
                + (y_node - self.load_centroid[1]) ** 2
            )

            self.nodes.distance_to_load_center.loc[node_index] = distance

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
                if self.nodes.node_type.loc[label] == "pole"
            ],
            axis=0,
        )

    def find_index_longest_distribution_link(self, max_distance_dist_links):
        # First select the distribution links from the entire links.
        distribution_links = self.links[self.links["link_type"] == "distribution"]

        # Find the links longer than two times of the allowed distance
        critical_link = distribution_links[
            distribution_links["length"] > max_distance_dist_links
        ]

        return list(critical_link.index)
        # if critical_link.length[0] > max_distance_dist_links:
        #     return critical_link.index[0]
        # else:
        #     return ""

        # for index in distribution_links.index:
        #     if distribution_links.at[index, "length"] > 2 * max_distance_dist_links:
        #         n_long_links += 1
        #         index_long_link = index

        # # Only when there is ONE very long connection in the gtid, the link
        # # label will be given to the optimizer to be ignored. Otherwise, an
        # # additional pole will be added to the grid.
        # if n_long_links == 1:
        #     return index_long_link
        # elif n_long_links > 1:
        #     return "many_long_links"
        # else:
        #     return ""

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
                x = x_from + np.sign(
                    x_to - x_from
                ) * i * length_smaller_links * np.sqrt(1 / (1 + slope**2))
                y = y_from + np.sign(y_to - y_from) * i * length_smaller_links * abs(
                    slope
                ) * np.sqrt(1 / (1 + slope**2))

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
        surface_area=0,
        peak_demand=0,  # FIXME: must be read automatically
        average_consumption=0,  # FIXME: must be read automatically
        distance_to_load_center=0,
        is_connected=True,
        how_added="automatic",
        type_fixed=False,
        segment="0",
        cluster_label=0,
        allocation_capacity=0,
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
        self.nodes.at[label, "surface_area"] = surface_area
        self.nodes.at[label, "peak_demand"] = peak_demand
        self.nodes.at[label, "average_consumption"] = average_consumption
        self.nodes.at[label, "distance_to_load_center"] = distance_to_load_center
        self.nodes.at[label, "is_connected"] = is_connected
        self.nodes.at[label, "how_added"] = how_added
        self.nodes.at[label, "type_fixed"] = type_fixed
        self.nodes.at[label, "segment"] = segment
        self.nodes.at[label, "cluster_label"] = cluster_label

    def remove_node(self, node_label):
        """
        This function removes the node with a given `node_label`.

        Parameter
        ---------
        node_label: str
            Node to be removed from the grid

        Notes
        -----
        If the `node_label` doesn't correspond to any node in the grid, the
        function raises a warning.
        """
        node_label = str(node_label)
        if node_label in self.nodes.index:
            self.nodes.at[node_label, "is_connected"] = False
        else:
            raise Warning(
                f"The node label given as input ('{node_label}') "
                + "doesn't exist in the grid"
            )

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
        Returns only the 'pole' nodes from the grid.

        Returns
        ------
        class:`pandas.core.frame.DataFrame`
            filtered pandas dataframe containing all 'pole' nodes
        """
        return self.nodes[self.nodes["node_type"] == "pole"]

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

    def remove_link(self, index):
        """
        Removes one link from the grid.
        """
        self.links = self.get_links().drop(index, axis=0)

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
        if (
            self.nodes.node_type.loc[label_node_from]
            and self.nodes.node_type.loc[label_node_to]
        ) == "pole":
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
            for node_index in self.poles().index:
                lon, lat = p(
                    self.nodes.x.loc[node_index] + self.ref_node[0],
                    self.nodes.y.loc[node_index] + self.ref_node[1],
                    inverse=inverse,
                )
                self.nodes.at[node_index, "longitude"] = lon
                self.nodes.at[node_index, "latitude"] = lat

        else:
            for node_index in self.consumers().index:
                x, y = p(
                    self.nodes.longitude.loc[node_index],
                    self.nodes.latitude.loc[node_index],
                    inverse=inverse,
                )
                self.nodes.at[node_index, "x"] = x
                self.nodes.at[node_index, "y"] = y

            # store reference values for (x,y) to use later when converting (x,y) to (lon,lat)
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

        # get the number of poles, consumers and links fron the grid
        n_poles = self.poles().shape[0]
        n_mg_consumers = self.consumers()[
            self.consumers()["is_connected"] == True
        ].shape[0]
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

    # -------------------- GET METHODS -------------------- #

    def get_non_fixed_nodes(self):
        """
        Returns filtered _nodes DataFrame with only nodes with
        type_fixed value being 'False'.

        Returns
        ------
        class:`pandas.core.frame.DataFrame`
            Filtered DataFrame containing all nodes with 'type_fixed' == True

        """
        return self.nodes[self.nodes["type_fixed"] == False].copy()  # noqa: E712

    def get_meter_per_default_unit(self):
        """
        Returns the __meter_per_default_unit attribute of the grid.

        Returns
        ------
        class: float
        """
        return self.meter_per_default_unit

    def get_segment_pole_capacity(self, segment):
        """
        Returns the total capacity of all the poles in the segment.

        Parameters
        ----------
        segment:
            Label of the segment
        """

        return self.get_poles()[self.get_poles()["segment"] == segment][
            "allocation_capacity"
        ].sum()

    def get_total_pole_capacity(self):
        """
        Returns the total capacity of all the poles in the grid.

        """
        return self.get_poles()["allocation_capacity"].sum()

    def get_id(self):
        """
        Returns __id attribute of the grid.

        Returns
        ------
        str
            __id parameter of the grid
        """
        return self.id

    def get_connection_cable_price(self):
        """
        Returns __connection_cable_price attribute of the grid.

        Returns
        ------
        int
            __connection_cable_price parameter of the grid
        """
        return self.cost_connection_link_per_meter

    def get_distribution_cable_price(self):
        """
        Returns __distribution_cable_price attribute of the grid.

        Returns
        ------
        int
            __distribution_cable_price parameter of the grid
        """
        return self.cost_pole_link_per_meter

    def get_price_pole(self):
        """
        Returns __price_pole attribute of the grid.

        Returns
        ------
        int
            __price_pole parameter of the grid
        """
        return self.cost_pole

    def get_price_consumer(self):
        """
        Returns __price_consumer attribute of the grid.

        Returns
        ------
        int
            __price_consumer parameter of the grid
        """
        return self.cost_connection

    # ------------------ FEATURE METHODS ------------------ #

    def does_link_exist(self, label_node1, label_node2):
        """
        This method returns True if there is a link bewteen the two node
        indices given as input, otherwise returns False.

        Parameters
        ----------
        label_node1: str
            Label of the first node
        label_node2: str
            Label of the second node

        Returns
        -------
        If there exists a link between the two nodes given as input, 'True' is
        returned, otherwise False is returned
        """

        if (
            self.get_links()[
                (self.get_links()["from"] == label_node1)
                & (self.get_links()["to"] == label_node2)
            ].shape[0]
            > 0
        ):
            return True
        elif (
            self.get_links()[
                (self.get_links()["from"] == label_node2)
                & (self.get_links()["to"] == label_node1)
            ].shape[0]
            > 0
        ):
            return True
        else:
            return False

    def is_pole_capacity_constraint_too_strong(self):
        """
        This methods returns wheter or not pole capacity constraint prevents
        from connecting all consumers to poles.

        Returns
        ------
            If number of consumers is greater than the sum of the respective
            segment's poles capacity, True is returned. Otherwise, False is
            returned.
        Note
        ----
            If all poles in the grid have a an allocation_capacity equals
            to 0, the allocation capacity is by default unrestricted and an
            arbitrary number of nodes can be assigned to each pole.
        """
        # If the sum of the allocation_capacity of the poles is 0, capacity is
        # by default unrestricted
        if self.get_poles()["allocation_capacity"].sum() == 0:
            return False

        is_capacity_constraint_too_strong = False
        for segment in self.get_nodes()["segment"].unique():
            if (
                self.consumers()[self.consumers()["segment"] == segment].shape[0]
                > self.get_poles()[self.get_poles()["segment"] == segment][
                    "allocation_capacity"
                ].sum()
            ):
                is_capacity_constraint_too_strong = True

        return is_capacity_constraint_too_strong

    def number_of_poles_required_to_meet_allocation_capacity_constraint(self):
        """This function computes the number of poles with defauld capacity
        required to meet allocation capacity constraint.

        Output
        ------
            (int):
                Number of poles with default capacity required to meet
                allocation capacity constraint.
        """
        # handle case where poles are uncapcitated
        if self.get_default_pole_capacity() == 0:
            return 1

        return int(
            np.ceil(self.get_nodes().shape[0] / (1 * self.get_default_pole_capacity()))
        )

    def is_segment_spanning_tree(self, segment):
        """
        This methods checks wheter or not the given nodes from the segment
        is spanning (i.e. buiding a connected graph).

        Parameters:
            str
                Segment index

        Returns:
            bool
                If the segment is spanning, True is returned.
                Otherwise False is returned
        """

        node_0 = self.get_nodes()[self.get_nodes()["segment"] == segment].index[0]

        nodes_in_spanning_tree = [node_0]
        loop_in_graph = False

        # find all neighbors of node_0
        neighbors_of_node_0 = []
        for index_neighbour in self.get_links()[(self.get_links()["from"] == node_0)][
            "to"
        ]:
            neighbors_of_node_0.append(index_neighbour)
        for index_neighbour in self.get_links()[(self.get_links()["to"] == node_0)][
            "from"
        ]:
            neighbors_of_node_0.append(index_neighbour)

        for neighbor in neighbors_of_node_0:
            if not loop_in_graph:
                self.loop_detection_iteration(
                    segment, nodes_in_spanning_tree, loop_in_graph, node_0, neighbor
                )
        segment_size = self.get_nodes()[self.get_nodes()["segment"] == segment].shape[0]
        return len(nodes_in_spanning_tree) == segment_size

    def loop_detection_iteration(
        self,
        segment,
        nodes_in_spanning_tree,
        loop_in_graph,
        previous_node,
        current_node,
    ):
        """
        This method is a helping method used in the function
        is_segment_spanning_tree to explore graph and discover wheter or not
        the segment is a connected tree.

        Parameters
        ----------
        segment: str
            Segment index
        nodes_in_spanning_tree: list
            List of the indices of the nodes that have been explored already
            and are in the spanning tree.
        loop_in_graph: bool
            Indicates whether or not a loop has been found in the graph
            This parameter is then passed to the calling function until it
            reaches the is_segment_spanning_tree function.
        previous_node: str
            Index of the node that came before in the iteration. This
            parameter is used to ensure that the exploration of the graph goes
            forward (except if there is a loop).
        current_node: str
            Index of the next node in the iteration.
        """

        if current_node in nodes_in_spanning_tree:
            loop_in_graph = True

        else:
            nodes_in_spanning_tree.append(current_node)
            neighbors_of_current_nodes = []
            for index_neighbour in self.get_links()[
                (self.get_links()["from"] == current_node)
            ]["to"]:
                if index_neighbour != previous_node:
                    neighbors_of_current_nodes.append(index_neighbour)
            for index_neighbour in self.get_links()[
                (self.get_links()["to"] == current_node)
            ]["from"]:
                if index_neighbour != previous_node:
                    neighbors_of_current_nodes.append(index_neighbour)

            for next_node in neighbors_of_current_nodes:
                self.loop_detection_iteration(
                    segment,
                    nodes_in_spanning_tree,
                    loop_in_graph,
                    current_node,
                    next_node,
                )

    # ------------------- SET METHODS --------------------- #

    def set_nodes(self, nodes):
        """
        Set grid's _nodes attibute to nodes parameter.

        Parameters
        ----------
        nodes : :class:`pandas.core.frame.DataFrame`
            node DataFrame (pandas) to set as Grid._nodes attribute.
        """
        self.nodes = nodes.copy()

    def set_links(self, links):
        """
        Set grid's _links attibute to links parameter.

        Parameters
        ----------
        links : :class:`pandas.core.frame.DataFrame`
            node DataFrame (pandas) to set as Grid._links attribute.
        """
        self.links = links.copy()

    # -------------- MANIPULATE NODES --------------- #
    def flip_node(self, node_label):
        """
        Switch the node_type of a node i.e. if node_type is 'pole',
        change it to 'consumer', if node_type is 'consumer', change
        it to 'pole'.

        Parameters
        ----------
        node_label: str
            label of the node.
        """

        if not self.nodes["type_fixed"][node_label]:
            if self.nodes["node_type"][node_label] == "pole":
                self.set_node_type(node_label=node_label, node_type="consumer")
                self.set_pole_capacity(str(node_label), 0)
            elif self.nodes["node_type"][node_label] == "consumer":
                self.set_node_type(node_label=node_label, node_type="pole")
                self.set_pole_capacity(str(node_label), self.pole_max_connection)

    def flip_random_node(self):
        """
        This function picks a node uniformly at random and flips its
        'node_type' (i.e. if node_type is pole, change it to
        consumer, if node_type is consumer, change it to pole).
        """
        # First be sure that the node dataframe is not empty
        if self.get_non_fixed_nodes().shape[0] > 0:  # noqa: E712
            randomly_selected_node_label = (
                self.get_non_fixed_nodes()[
                    self.get_non_fixed_nodes()["node_type"] != "powerhub"
                ]
                .sample(n=1)
                .index[0]
            )
            self.flip_node(randomly_selected_node_label)

    def swap_random(self, swap_option="random"):
        """
        This method picks a pole uniformly at random and, swap it
        house state with a consumer selected according to the
        swap_option parameter (ie. the 'node_type' of the picked
        pole is changed to 'consumer' and the 'node_type' of the
        selected consumer is set to 'pole).

        Parameters
        ----------
        swap_option: :class:`bool`
            If parameter is 'nearest_neighbour', the consumer that is picked
            is necessarily the one that is the clostest to the picked
            pole. If parameter is 'random', the consumer to be swaped
            with the pole is selected uniformly at random.
        """

        # Make sure that the grid contains at least one pole and consumer
        if (
            self.get_non_fixed_nodes()[
                self.get_non_fixed_nodes()["node_type"] == "pole"
            ].shape[0]
            > 0
            and self.get_non_fixed_nodes()[
                self.get_non_fixed_nodes()["node_type"] == "consumer"
            ].shape[0]
            > 0
        ):

            # Pick a pole uniformly at random among the ones not fixed
            randomly_selected_pole_label = (
                self.get_non_fixed_nodes()[
                    self.get_non_fixed_nodes()["node_type"] == "pole"
                ]
                .sample(n=1)
                .index[0]
            )

            # If swap_option is 'nearest_neighbour', find nearest consumer
            # and flip its node_type
            if swap_option == "nearest_neighbour":
                # Define first consumer of the nodes Dataframe before looping
                # to finde the nearest to the picked pole and save
                # distance
                selected_consumer_label = self.get_non_fixed_nodes()[
                    self.get_non_fixed_nodes()["node_type"] == "consumer"
                ].index[0]

                dist_to_selected_consumer = self.distance_between_nodes(
                    randomly_selected_pole_label, selected_consumer_label
                )
                # Loop over all consumers to find the one that is the nearest
                # to the pole
                for consumer_label in self.get_non_fixed_nodes()[
                    self.get_non_fixed_nodes()["node_type"] == "consumer"
                ].index:
                    if (
                        self.distance_between_nodes(
                            randomly_selected_pole_label, consumer_label
                        )
                        < dist_to_selected_consumer
                    ):
                        selected_consumer_label = consumer_label
                        dist_to_selected_consumer = self.distance_between_nodes(
                            randomly_selected_pole_label, selected_consumer_label
                        )
            else:
                selected_consumer_label = (
                    self.get_non_fixed_nodes()[
                        self.get_non_fixed_nodes()["node_type"] == "consumer"
                    ]
                    .sample(n=1)
                    .index[0]
                )

            self.flip_node(randomly_selected_pole_label)
            self.flip_node(selected_consumer_label)

    def set_all_node_type_to_consumers(self):
        """ "
        This method sets the node_type to 'consumer' for all nodes with
        type_fixed == False.
        """

        for label in self.get_non_fixed_nodes()[
            (self.nodes["node_type"] != "powerhub")
        ].index:
            self.set_node_type(label, "consumer")

    def set_all_node_type_to_poles(self):
        """ "
        This method sets the node_type to 'pole' for all nodes with
        type_fixed == False.
        """

        for label in self.nodes[self.nodes["node_type"] != "powerhub"].index:
            if not self.get_nodes()["type_fixed"][label]:
                self.set_node_type(label, "pole")

    def set_node_type(self, node_label, node_type):
        """
        This method set the node type of a given node to the value
        given as parameter.

        Parameter
        ---------
            node_label: str
                Label of the node contained in grid.
            node_type: str
                value the 'node_type' of the given node is set to.
        """
        if not self.get_nodes()["type_fixed"][node_label]:
            self.nodes.at[node_label, "node_type"] = node_type
            if node_type == "pole" or node_type == "powerhub":
                self.nodes.at[
                    node_label, "allocation_capacity"
                ] = self.pole_max_connection
            elif node_type == "consumer":
                self.nodes.at[node_label, "allocation_capacity"] = 0

    def set_node_type_randomly(self, probability_for_pole):
        """ "
        This method sets the node_type of each node to pole with a
        probability probability_for_pole, the rest are being set to
        consumers.

        Parameters
        ----------
            probability_for_pole: float
                Probabilty to assign each node to node_type value 'pole'.
        """

        for label in self.nodes.index:
            if np.random.rand() < probability_for_pole:
                self.set_node_type(node_label=label, node_type="pole")
            else:
                self.set_node_type(node_label=label, node_type="consumer")

    def set_segment(self, node_label, segment):
        """This method assigns the segment attribute of the node corresponding
        to node_label to the value of segment.

        Parameters
        ----------
        node_label: str
            Label of the node
        segment: str
            Label of the segment the node should be assigned to.
        Notes
        -----
            If the node label doesn't correspond to any node in the grid,
            method does nothing.
        """
        if node_label in self.nodes.index:
            segment = str(segment)
            self.nodes.at[str(node_label), "segment"] = segment

    def set_type_fixed(self, node_label, type_to_set):
        """
        Set the type_fixed of the selected node to the value of type_to_set.

        Parameters
        ----------
        node_label: str
            label of the node.
        type_to_set: :class:`bool`
            value of the type_fixed of the node should be set to.
        Note
        ----
        The node_type of the nodes with type_fixed is True shouldn't not be
        changed.
        """
        if self.nodes.shape[0] > 0:
            self.nodes.at[str(node_label), "type_fixed"] = type_to_set

    def set_pole_capacity(self, pole_label, allocation_capacity):
        """
        This method sets the allocation capacity of a pole to the value given
        by the allocation_capacity parameter. If the node is not a pole, the
        method doesn't do anything.

        Parameters
        ----------
        pole_label: str
            Label of the pole.
        allocation_capacity: int
            Value the allocation_capacity of the pole is assigned to.
        """
        if pole_label in self.get_poles().index and type(allocation_capacity) == int:
            self.nodes.at[str(pole_label), "allocation_capacity"] = allocation_capacity

    def set_default_pole_capacity(self, default_pole_capacity):
        """
        Set grid's _default_pole_capacity attibute to default_pole_capacity parameter.

        Parameters
        ----------
        links (int):
            Value to set to default pole capacity.
        """
        self.pole_max_connection = default_pole_capacity

    def shift_node(self, node, delta_x: float, delta_y: float):
        """
        +++ ok +++

        This method increments the 'x_coordinate' value by delta_x and the
        'y_coordinate' value by delta_y for the node given as parameter.

        Parameters
        ----------
            node: str
                Index of the node that should be moved.
            delta_x: float
                The value that must be added to the 'x_coordinate' of the given node.
            delta_y: int
                The value that must be added to the 'y_coordinate' of the given node.
        """

        self.nodes.x[node] += delta_x
        self.nodes.y[node] += delta_y

    # ----------------------- MANIPULATE LINKS ------------------------ #

    def remove_link_2(self, node1, node2):
        """
        This method removes, if it exists, the link between the two nodes
        given as parameter.

        Parameters
        ----------
        node1: str
            Label of one of the nodes connected by the link.
        node2: str
            Label of the other node connected by the link.
        """
        (label_node_from, label_node_to) = sorted([node1, node2])
        link_label = f"({label_node_from}, {label_node_to})"
        if link_label in self.get_links().index:
            self.links = self.links.drop(link_label, axis=0)
        else:
            raise Warning(
                f"The link between {node1} and {node2} cannot be  "
                + "removed since the two nodes are not connected"
            )

    def clear_distribution_links(self):
        """Removes all the distribution links from the grid$."""
        self.links = self.links[self.links["type"] != "distribution"]

    def clear_connection_links(self):
        """Removes all the connection links from the grid."""
        self.links = self.links[self.links["type"] != "connection"]

    # ----------------- COMPUTE DISTANCE BETWEEN NODES -----------------#

    def get_cable_distance_from_consumers_to_powerhub(self):
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
        for index_powerhub in self.get_nodes()[
            self.get_nodes()["node_type"] == "powerhub"
        ].index:

            distance_df.loc[index_powerhub] = [0, 0, index_powerhub]

            # this list gathers the index of all nodes that are directly
            # connected with a link to the powerhub
            node_next_neighbours = []
            # add all nodes connected to the pole to the list
            for next_node in self.get_links()[
                (self.get_links()["from"] == index_powerhub)
            ]["to"]:
                if (
                    next_node not in node_next_neighbours
                    and next_node not in distance_df.index
                ):
                    node_next_neighbours.append(next_node)
            for next_node in self.get_links()[
                (self.get_links()["to"] == index_powerhub)
            ]["from"]:
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

    def measure_distance_for_next_node(
        self, node_n_minus_1, node_n, distance_df, index_powerhub
    ):
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
                (self.get_links()["from"] == node_n_minus_1)
                & (self.get_links()["to"] == str(node_n))
            ].shape[0]
            == 1
        ):
            index_link_between_nodes = self.get_links()[
                (self.get_links()["from"] == node_n_minus_1)
                & (self.get_links()["to"] == str(node_n))
            ].index[0]

        elif (
            self.get_links()[
                (self.get_links()["from"] == node_n)
                & (self.get_links()["to"] == node_n_minus_1)
            ].shape[0]
            == 1
        ):
            index_link_between_nodes = self.get_links()[
                (self.get_links()["from"] == node_n)
                & (self.get_links()["to"] == node_n_minus_1)
            ].index[0]

        # check what type the link is to know distiguish of the cable types
        # in the datafram
        if self.get_links()["type"][index_link_between_nodes] == "distribution":
            distance_df.loc[node_n] = [
                distance_df["distribution cable [m]"][node_n_minus_1]
                + self.distance_between_nodes(node_n_minus_1, node_n),
                0,
                index_powerhub,
            ]
        elif self.get_links()["type"][index_link_between_nodes] == "connection":
            distance_df.loc[node_n] = [
                distance_df["distribution cable [m]"][node_n_minus_1],
                self.distance_between_nodes(node_n_minus_1, node_n),
                index_powerhub,
            ]

        # Call function for all the nodes that were not measured yet
        node_next_neighbours = []
        for node_next_neighbour in self.get_links()[
            (self.get_links()["from"] == node_n)
        ]["to"]:
            if (
                node_next_neighbour not in node_next_neighbours
                and node_next_neighbour not in distance_df.index
            ):
                node_next_neighbours.append(node_next_neighbour)
        for node_next_neighbour in self.get_links()[(self.get_links()["to"] == node_n)][
            "from"
        ]:
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
            The cable resistance R_i is computed as follow
            R_i =  rho_i * 2* d_i / (i_cable_section)
            where i represent the cable type, rho the cable electric
            resistivity (self.distribution_cable_resistivity),
            d the cable distance and i_cable_section the section of the cable.
            The voltage drop is computed using Ohm's law
            U = R * I where U is the tension (here corresponding to the
            voltage drop), R the resistance and I the current.
        """
        voltage_drop_df = self.get_cable_distance_from_consumers_to_powerhub()

        voltage_drop_df["distribution cable resistance [Ω]"] = (
            self.distribution_cable_resistivity
            * 2
            * voltage_drop_df["distribution cable [m]"]
            / self.distribution_cable_section
        )

        voltage_drop_df["connection cable resistance [Ω]"] = (
            self.connection_cable_resistivity
            * 2
            * voltage_drop_df["connection cable [m]"]
            / self.connection_cable_section
        )

        voltage_drop_df["voltage drop [V]"] = (
            voltage_drop_df["distribution cable resistance [Ω]"] * self.max_current
        ) + (voltage_drop_df["connection cable resistance [Ω]"] * self.max_current)

        voltage_drop_df["voltage drop fraction [%]"] = (
            100 * voltage_drop_df["voltage drop [V]"] / self.voltage
        )

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
