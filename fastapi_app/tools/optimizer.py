from __future__ import division
import pandas as pd
from fastapi_app.io.db import queries
import fastapi_app.io.db.models as models


async def check_data_availability(user_id, project_id):
    project_setup = await queries.get_model_instance(models.ProjectSetup, user_id, project_id)
    if project_setup is None:
        return False, '/project_setup/?project_id=' + str(project_id)
    nodes = await queries.get_model_instance(models.Nodes, user_id, project_id)
    nodes_df = pd.read_json(nodes.data) if nodes is not None else None
    if nodes_df is None or nodes_df.empty or nodes_df[nodes_df['node_type'] == 'consumer'].index.__len__() == 0:
        return False, '/consumer_selection/?project_id=' + str(project_id)
    demand_opt_dict = await queries.get_model_instance(models.Demand, user_id, project_id)
    if demand_opt_dict is not None:
        demand_opt_dict = demand_opt_dict.to_dict()
    if demand_opt_dict is None or demand_opt_dict['household_option'] is None:
        return False, '/demand_estimation/?project_id=' + str(project_id)
    grid_design = await queries.get_df(models.GridDesign, user_id, project_id, is_timeseries=False)
    if grid_design is None or grid_design.empty or pd.isna(grid_design['pole_lifetime'].iat[0]):
        return False, '/grid_design/?project_id=' + str(project_id)
    energy_system_design = await queries.get_energy_system_design(user_id, project_id)
    if grid_design is None or energy_system_design['battery']['parameters']['c_rate_in'] is None:
        return False, '/energy_system_design/?project_id=' + str(project_id)
    else:
        return True, None


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







