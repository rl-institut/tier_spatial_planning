from __future__ import division
import os
import time
from datetime import datetime, timedelta
import pandas as pd
from pyomo import environ as po
from fastapi_app.db import async_queries, sync_queries, sync_inserts, queries_demand
from fastapi_app import config
import fastapi_app.db.sa_tables as models
from fastapi_app.tools.energy_system_model import EnergySystemOptimizer
from fastapi_app.tools import energy_system_model
from fastapi_app.tools.error_logger import logger as error_logger
from fastapi_app.tools.grid_obj import Grid
from fastapi_app.tools.grid_model import GridOptimizer
from fastapi_app.tools.mails import send_mail
from fastapi_app.tools.solar_potential import get_dc_feed_in_sync_db_query


async def check_data_availability(user_id, project_id):
    project_setup = await async_queries.get_model_instance(models.ProjectSetup, user_id, project_id)
    if project_setup is None:
        return False, '/project_setup/?project_id=' + str(project_id)
    nodes = await async_queries.get_model_instance(models.Nodes, user_id, project_id)
    nodes_df = pd.read_json(nodes.data) if nodes is not None else None
    if nodes_df is None or nodes_df.empty or nodes_df[nodes_df['node_type'] == 'consumer'].index.__len__() == 0:
        return False, '/consumer_selection/?project_id=' + str(project_id)
    demand_opt_dict = await async_queries.get_model_instance(models.Demand, user_id, project_id)
    if demand_opt_dict is not None:
        demand_opt_dict = demand_opt_dict.to_dict()
    if demand_opt_dict is None or demand_opt_dict['household_option'] is None:
        return False, '/demand_estimation/?project_id=' + str(project_id)
    grid_design = await async_queries.get_df(models.GridDesign, user_id, project_id, is_timeseries=False)
    if grid_design is None or grid_design.empty or pd.isna(grid_design['pole_lifetime'].iat[0]):
        return False, '/grid_design/?project_id=' + str(project_id)
    energy_system_design = await async_queries.get_energy_system_design(user_id, project_id)
    if grid_design is None or energy_system_design['battery']['parameters']['c_rate_in'] is None:
        return False, '/energy_system_design/?project_id=' + str(project_id)
    else:
        return True, None


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
        opt = GridOptimizer(start_date=df.loc[0, "start_date"],
                            n_days=df.loc[0, "n_days"],
                            project_lifetime=df.loc[0, "project_lifetime"],
                            wacc=df.loc[0, "interest_rate"] / 100,
                            tax=0, )
        nodes = sync_queries.get_model_instance(models.Nodes, user_id, project_id)
        nodes = pd.read_json(nodes.data)
        nodes['is_connected'] = True
        nodes.loc[nodes['shs_options'] == 2, 'is_connected'] = False
        nodes.index = nodes.index.astype(str)
        nodes = nodes[nodes['node_type'].isin(['consumer', 'power-house'])]
        power_house = nodes.loc[nodes['node_type'] == 'power-house']
        if power_house.index.__len__() > 0 and power_house['how_added'].iat[0] != 'manual':
            nodes = nodes.drop(index=power_house.index)
            power_house = None
        elif power_house.index.__len__() == 0:
            power_house = None
        if len(nodes) == 0:
            return {"code": "success", "message": "Empty grid cannot be optimized!"}

        epc_distribution_cable = opt.calc_epc("distribution_cable_capex", "distribution_cable_lifetime", df)
        epc_connection_cable = opt.calc_epc("connection_cable_capex", "connection_cable_lifetime", df)
        epc_connection = opt.calc_epc("mg_connection_cost", "project", df)
        epc_pole = opt.calc_epc("pole_capex", "pole_lifetime", df)

        # This part calculated the total consumption of the community for the
        # selected time period.
        start_date_obj = opt.start_date
        start_datetime = datetime.combine(start_date_obj.date(), start_date_obj.time())
        end_datetime = start_datetime + timedelta(days=int(opt.n_days))

        demand_opt_dict = sync_queries.get_model_instance(models.Demand, user_id, project_id).to_dict()
        demand_full_year = queries_demand.get_demand_time_series(nodes, demand_opt_dict).to_frame('Demand')
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

        # make sure that the new grid object is empty before adding nodes to it
        grid.clear_nodes()
        grid.clear_all_links()

        # exclude solar-home-systems and poles from the grid optimization
        grid.nodes = nodes

        # convert all (long,lat) coordinates to (x,y) coordinates and update
        # the Grid object, which is necessary for the GridOptimizer
        grid.convert_lonlat_xy()

        # in case the grid contains 'poles' from the previous optimization
        # they must be removed, becasue the grid_optimizer will calculate
        # new locations for poles considering the newly added nodes
        grid.clear_poles()

        # Find the number of SHS consumers (temporarily)
        n_total_consumers = grid.nodes.index.__len__()
        n_shs_consumers = nodes[nodes["is_connected"] == False].index.__len__()
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

        # Find the connection links in the network with lengths greater than the
        # maximum allowed length for `connection` cables, specified by the user.
        long_links = grid.find_index_longest_distribution_link(max_distance_dist_links=distribution_cable_max_length)

        # Add poles to the identified long `distribution` links, so that the
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
        iter = 2 if power_house is None else 1
        for i in range(iter):
            if power_house is None and i == 0:
                grid.select_location_of_power_house()
            grid.set_direction_of_links()
            grid.allocate_poles_to_branches()
            grid.allocate_subbranches_to_branches()
            grid.label_branch_of_consumers()
            grid.determine_cost_per_pole()
            grid.connection_cost_per_consumer()
            grid.determine_costs_per_branch()
            # ToDo: demand of each consumer should be calculated here.
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
        nodes = grid.nodes.reset_index(drop=True)
        nodes.drop(labels=["x", "y", "cluster_label", "type_fixed", "n_connection_links", "n_distribution_links",
                           "cost_per_pole", "branch", "parent_branch", "total_grid_cost_per_consumer_per_a",
                           "connection_cost_per_consumer", 'cost_per_branch', 'distribution_cost_per_branch',
                           'yearly_consumption'],
                   axis=1,
                   inplace=True)
        sync_inserts.update_nodes_and_links(True, False, nodes, user_id, project_id, replace=True)
        links = grid.links.reset_index(drop=True)
        links.drop(labels=["x_from", "y_from", "x_to", "y_to", "n_consumers", "total_power", "from_node", "to_node"],
                   axis=1,
                   inplace=True)
        sync_inserts.update_nodes_and_links(False, True, links.to_dict(), user_id, project_id, replace=True)
        end_execution_time = time.monotonic()
        results = models.Results()
        results.n_consumers = len(grid.consumers())
        results.n_shs_consumers = nodes[nodes["is_connected"] == False].index.__len__()
        results.n_poles = len(grid.poles())
        results.length_distribution_cable = int(grid.links[grid.links.link_type == "distribution"]["length"].sum())
        results.length_connection_cable = int(grid.links[grid.links.link_type == "connection"]["length"].sum())
        results.cost_grid = int(grid.cost()) if grid.links.index.__len__() > 0 else 0
        results.cost_shs = int(cost_shs)
        results.time_grid_design = round(end_execution_time - start_execution_time, 1)
        results.n_distribution_links = int(grid.links[grid.links["link_type"] == "distribution"].shape[0])
        results.n_connection_links = int(grid.links[grid.links["link_type"] == "connection"].shape[0])


        df = results.to_df()
        sync_inserts.insert_results_df(df, user_id, project_id)
    except Exception as exc:
        user_name = 'user with user_id: {}'.format(user_id)
        error_logger.error_log(exc, 'no request', user_name)
        raise exc


def optimize_energy_system(user_id, project_id):
    try:
        print('start es opt')
        start_execution_time = time.monotonic()
        df = sync_queries.get_input_df(user_id, project_id)
        energy_system_design = sync_queries.get_energy_system_design(user_id, project_id)
        solver = 'gurobi' if po.SolverFactory('gurobi').available() else 'cbc'
        if solver == 'cbc':
            energy_system_design['diesel_genset']['settings']['offset'] = False
        nodes = sync_queries.get_model_instance(models.Nodes, user_id, project_id)
        nodes = pd.read_json(nodes.data)
        num_households = len(nodes[(nodes['consumer_type'] == 'household') &
                                   (nodes['is_connected'] == True)].index)
        if num_households == 0:
            return False
        if not nodes[nodes['consumer_type'] == 'power_house'].empty:
            lat, lon = nodes[nodes['consumer_type'] == 'power_house']['latitude', 'longitude'].to_list()
        else:
            lat, lon = nodes[['latitude', 'longitude']].mean().to_list()
        n_days = min(df.loc[0, "n_days"], int(os.environ.get('MAX_DAYS', 365)))
        start = pd.to_datetime(df.loc[0, "start_date"])
        end = start + timedelta(days=int(n_days))
        solar_potential_df = get_dc_feed_in_sync_db_query(lat, lon, start, end)
        demand_opt_dict = sync_queries.get_model_instance(models.Demand, user_id, project_id).to_dict()
        demand_full_year = queries_demand.get_demand_time_series(nodes, demand_opt_dict).to_frame('Demand')
        ensys_opt = EnergySystemOptimizer(
            start_date=df.loc[0, "start_date"],
            n_days=n_days,
            project_lifetime=df.loc[0, "project_lifetime"],
            wacc=df.loc[0, "interest_rate"] / 100,
            tax=0,
            solar_potential=solar_potential_df,
            demand=demand_full_year,
            solver=solver,
            pv=energy_system_design['pv'],
            diesel_genset=energy_system_design['diesel_genset'],
            battery=energy_system_design['battery'],
            inverter=energy_system_design['inverter'],
            rectifier=energy_system_design['rectifier'],
            shortage=energy_system_design['shortage'], )
        ensys_opt.optimize_energy_system()
        end_execution_time = time.monotonic()
        if ensys_opt.model.solutions.__len__() == 0:
            if ensys_opt.infeasible is True:
                df = sync_queries.get_df(models.Results, user_id, project_id)
                df.loc[0, "infeasible"] = ensys_opt.infeasible
                sync_inserts.insert_results_df(df, user_id, project_id)
            return False
        df, emissions, co2_emission_factor = energy_system_model.get_emissions(ensys_opt, user_id, project_id)
        sync_inserts.merge_model(emissions)
        co2_savings = df.loc[:, "co2_savings"].max()
        df = sync_queries.get_df(models.Results, user_id, project_id)
        grid_input_parameter = sync_queries.get_input_df(user_id, project_id)
        links = sync_queries.get_model_instance(models.Links, user_id, project_id)
        df = energy_system_model.get_results_df(ensys_opt,
                       df,
                       n_days,
                       grid_input_parameter,
                       demand_full_year,
                       co2_savings,
                       nodes,
                       links,
                       num_households,
                       end_execution_time,
                       start_execution_time,
                       energy_system_design,
                       co2_emission_factor)
        sync_inserts.insert_results_df(df, user_id, project_id)
        energy_flow = energy_system_model.get_energy_flow(ensys_opt, user_id, project_id)
        sync_inserts.merge_model(energy_flow)
        demand_coverage = energy_system_model.get_demand_coverage(ensys_opt, user_id, project_id)
        sync_inserts.merge_model(demand_coverage)
        demand_curve = energy_system_model.get_demand_curve(ensys_opt, user_id, project_id)
        sync_inserts.merge_model(demand_curve)
        project_setup = sync_queries.get_model_instance(models.ProjectSetup, user_id, project_id)
        project_setup.status = "finished"
        if project_setup.email_notification is True:
            user = sync_queries.get_user_by_id(user_id)
            subject = "PeopleSun: Model Calculation finished"
            msg = "The calculation of your optimization model is finished. You can view the results at: " \
                  "\n\n{}/simulation_results?project_id={}\n".format(config.DOMAIN, project_id)
            send_mail(user.email, msg, subject=subject)
        project_setup.email_notification = False
        sync_inserts.merge_model(project_setup)
        return True
    except Exception as exc:
        user_name = 'user with user_id: {}'.format(user_id)
        error_logger.error_log(exc, 'no request', user_name)
        raise exc
