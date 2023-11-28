import time
from datetime import datetime, timedelta
import pandas as pd
from fastapi_app.db import sync_queries, sync_inserts, queries_demand
from fastapi_app.db import sa_tables
from fastapi_app.helper.error_logger import logger as error_logger
from fastapi_app.models.grid_obj import Grid
from fastapi_app.models.grid_optimizer import GridOptimizer


def optimize_grid(user_id, project_id):
    try:
        print('start grid opt')
        # Grab Currrent Time Before Running the Code
        project_setup = sync_queries.get_project_setup_of_user(user_id, project_id)
        project_setup.status = "in progress"
        sync_inserts.merge_model(project_setup)
        start_execution_time = time.monotonic()
        # create GridOptimizer object
        df = sync_queries.get_input_df(user_id, project_id)
        opt = GridOptimizer(user_id=user_id,
                            project_id=project_id,
                            start_datetime=df.loc[0, "start_date"],
                            n_days=df.loc[0, "n_days"],
                            project_lifetime=df.loc[0, "project_lifetime"],
                            wacc=df.loc[0, "interest_rate"] / 100,
                            tax=0, )
        nodes_df = sync_queries.get_model_instance(sa_tables.Nodes, user_id, project_id)
        nodes_df = pd.read_json(nodes_df.data)
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
        if len(nodes_df) == 0:
            return {"code": "success", "message": "Empty grid cannot be optimized!"}

        epc_distribution_cable = opt.calc_epc("distribution_cable_capex", "distribution_cable_lifetime", df)
        epc_connection_cable = opt.calc_epc("connection_cable_capex", "connection_cable_lifetime", df)
        epc_connection = opt.calc_epc("mg_connection_cost", "project", df)
        epc_pole = opt.calc_epc("pole_capex", "pole_lifetime", df)

        # This part calculated the total consumption of the community for the
        # selected time period.
        start_date_obj = opt.start_datetime
        start_datetime = datetime.combine(start_date_obj.date(), start_date_obj.time())
        end_datetime = start_datetime + timedelta(days=int(opt.n_days))

        demand_opt_dict = sync_queries.get_model_instance(sa_tables.Demand, user_id, project_id).to_dict()
        demand_full_year = queries_demand.get_demand_time_series(nodes_df, demand_opt_dict).to_frame('Demand')
        demand_full_year.index = pd.date_range(start=start_datetime, periods=len(demand_full_year), freq="H")

        # Then the demand for the selected time period given by the user will be
        # obtained.
        demand_selected_period = demand_full_year.Demand.loc[start_datetime:end_datetime]

        grid = Grid(
            epc_distribution_cable=epc_distribution_cable,
            epc_connection_cable=epc_connection_cable,
            epc_connection=epc_connection,
            epc_pole=epc_pole,
            pole_max_connection=df.loc[0, "pole_max_n_connections"],
            max_levelized_grid_cost=df.loc[0, "shs_max_grid_cost"])

        # make sure that the new grid object is empty before adding nodes_df to it
        grid.clear_nodes()
        grid.clear_all_links()

        # exclude solar-home-systems and poles from the grid optimization
        grid.nodes = nodes_df

        # convert all (long,lat) coordinates to (x,y) coordinates and update
        # the Grid object, which is necessary for the GridOptimizer
        grid.convert_lonlat_xy()

        # in case the grid contains 'poles' from the previous optimization
        # they must be removed, becasue the grid_optimizer will calculate
        # new locations for poles considering the newly added nodes_df
        grid.clear_poles()

        # Find the number of SHS consumers (temporarily)
        n_total_consumers = grid.nodes.index.__len__()
        n_shs_consumers = nodes_df[nodes_df["is_connected"] == False].index.__len__()
        n_grid_consumers = n_total_consumers - n_shs_consumers
        grid.nodes.sort_index(key=lambda x: x.astype("int64"), inplace=True)

        if power_house is not None:
            power_house_consumers = grid.connect_power_house_consumer_manually(df.loc[0, "connection_cable_max_length"])
            grid.placeholder_consumers_for_power_house()
        else:
            power_house_consumers = None

        n_poles = opt.find_opt_number_of_poles(grid, df.loc[0, "connection_cable_max_length"], n_grid_consumers)
        opt.determine_poles(grid=grid,
                            min_n_clusters=n_poles,
                            power_house_consumers=power_house_consumers,
                            power_house=power_house)
        distribution_cable_max_length = df.loc[0, "distribution_cable_max_length"]

        # Find the connection links_df in the network with lengths greater than the
        # maximum allowed length for `connection` cables, specified by the user.
        long_links = grid.find_index_longest_distribution_link(max_distance_dist_links=distribution_cable_max_length)

        # Add poles to the identified long `distribution` links_df, so that the
        # distance between all poles remains below the maximum allowed distance.
        grid.add_fixed_poles_on_long_links(long_links=long_links, max_allowed_distance=distribution_cable_max_length)

        # Update the (lon,lat) coordinates based on the newly inserted poles
        # which only have (x,y) coordinates.
        grid.convert_lonlat_xy(inverse=True)

        # Connect all poles together using the minimum spanning tree algorithm.
        opt.connect_grid_poles(grid, long_links=long_links)

        # Calculate distances of all poles from the load centroid.


        # Find the location of the power house.
        grid.add_number_of_distribution_and_connection_cables()
        n_iter = 2 if power_house is None else 1
        for i in range(n_iter):
            if power_house is None and i == 0:
                grid.select_location_of_power_house()
            grid.set_direction_of_links()
            grid.allocate_poles_to_branches()
            grid.allocate_subbranches_to_branches()
            grid.label_branch_of_consumers()
            grid.determine_cost_per_pole()
            grid.connection_cost_per_consumer()
            grid.determine_costs_per_branch()
            consumer_idxs = grid.nodes[grid.nodes['node_type'] == 'consumer'].index
            grid.nodes.loc[consumer_idxs, 'yearly_consumption'] = demand_selected_period.sum() / len(consumer_idxs)
            grid.determine_shs_consumers()
            if power_house is None and grid.links.index.__len__() > 0:
                old_power_house = grid.nodes[grid.nodes["node_type"] == 'power-house'].index[0]
                grid.select_location_of_power_house()
                new_power_house = grid.nodes[grid.nodes["node_type"] == 'power-house'].index[0]
                if old_power_house == new_power_house:
                    break
            else:
                break

        cost_shs =  0 #peak_demand_shs_consumers.sum()

        # get all poles obtained by the network relaxation method
        nodes_df = grid.nodes.reset_index(drop=True)
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
                nodes.id = user_id
                nodes.project_id = project_id
                nodes.data = nodes_df.reset_index(drop=True).to_json()
                sync_inserts.merge_model(nodes)
        links_df = grid.links.reset_index(drop=True)
        links_df.drop(labels=["x_from", "y_from", "x_to", "y_to", "n_consumers", "total_power", "from_node", "to_node"],
                   axis=1,
                   inplace=True)
        links_df.lat_from = links_df.lat_from.map(lambda x: "%.6f" % x)
        links_df.lon_from = links_df.lon_from.map(lambda x: "%.6f" % x)
        links_df.lat_to = links_df.lat_to.map(lambda x: "%.6f" % x)
        links_df.lon_to = links_df.lon_to.map(lambda x: "%.6f" % x)
        if len(df.index) != 0:
            links = sa_tables.Links()
            links.id = user_id
            links.project_id = project_id
            links.data = links_df.reset_index(drop=True).to_json()
            sync_inserts.merge_model(links)
        end_execution_time = time.monotonic()
        results = sa_tables.Results()
        results.n_consumers = len(grid.consumers())
        results.n_shs_consumers = nodes_df[nodes_df["is_connected"] == False].index.__len__()
        results.n_poles = len(grid.poles())
        results.length_distribution_cable = int(grid.links[grid.links.link_type == "distribution"]["length"].sum())
        results.length_connection_cable = int(grid.links[grid.links.link_type == "connection"]["length"].sum())
        results.cost_grid = int(grid.cost()) if grid.links.index.__len__() > 0 else 0
        results.cost_shs = int(cost_shs)
        results.time_grid_design = round(end_execution_time - start_execution_time, 1)
        results.n_distribution_links = int(grid.links[grid.links["link_type"] == "distribution"].shape[0])
        results.n_connection_links = int(grid.links[grid.links["link_type"] == "connection"].shape[0])
        results.id = user_id
        results.project_id = project_id
        sync_inserts.merge_model(results)
    except Exception as exc:
        user_name = 'user with user_id: {}'.format(user_id)
        error_logger.error_log(exc, 'no request', user_name)
        raise exc
