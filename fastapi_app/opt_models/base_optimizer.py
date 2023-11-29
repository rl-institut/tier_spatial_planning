import os
import pandas as pd
from fastapi_app.db import sync_queries, sa_tables
from fastapi_app.inputs import demand_estimation


class BaseOptimizer:
    """
    This is a general parent class for both grid and energy system optimizers
    """

    def __init__(
        self, user_id, project_id, ):
        self.project_setup = {k: v[0] if isinstance(v, tuple) and len(v) == 1 else v for k, v in
                         sync_queries.get_input_df(user_id, project_id).iloc[0].to_dict().items()}
        self.user_id=user_id
        self.project_id=project_id
        n_days = min(self.project_setup["n_days"], int(os.environ.get('MAX_DAYS', 365)))
        self.start_datetime = pd.to_datetime(self.project_setup["start_date"]).to_pydatetime()
        self.dt_index = pd.date_range(self.start_datetime,
                                      self.start_datetime + pd.to_timedelta(n_days, unit="D"),
                                      freq='H',
                                      closed='left')
        self.n_days = n_days
        self.project_lifetime = self.project_setup["project_lifetime"]
        self.wacc = self.project_setup["interest_rate"] / 100
        self.tax = 0
        self.crf = (self.wacc * (1 + self.wacc) ** self.project_lifetime) / \
                   ((1 + self.wacc) ** self.project_lifetime - 1)
        self.nodes = pd.read_json(sync_queries.get_model_instance(sa_tables.Nodes, self.user_id, self.project_id).data)
        demand_opt_dict = sync_queries.get_model_instance(sa_tables.Demand, user_id, project_id).to_dict()
        self.demand_full_year = demand_estimation.get_demand_time_series(self.nodes, demand_opt_dict).to_frame('Demand')
        self.demand = self.demand_full_year.loc[self.dt_index]['Demand'].copy()

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
            number_of_investments = int(round(self.project_lifetime / component_lifetime + 0.5))
        first_time_investment = capex_0 * (1 + self.tax)
        capex = first_time_investment
        for count_of_replacements in range(1, number_of_investments):
            if count_of_replacements * component_lifetime != self.project_lifetime:
                    capex += first_time_investment / ((1 + self.wacc) ** (count_of_replacements * component_lifetime))
        # Substraction of component value at end of life with last replacement (= number_of_investments - 1)
        # This part calculates the salvage costs
        if number_of_investments * component_lifetime > self.project_lifetime:
            last_investment = first_time_investment / (
                (1 + self.wacc) ** ((number_of_investments - 1) * component_lifetime)
            )
            linear_depreciation_last_investment = last_investment / component_lifetime
            capex = capex - linear_depreciation_last_investment * (
                number_of_investments * component_lifetime - self.project_lifetime
            ) / ((1 + self.wacc) ** self.project_lifetime)
        return capex
