import pandas as pd


class BaseOptimizer:
    """
    This is a general parent class for both grid and energy system optimizers
    """
    def __init__(
        self, user_id, project_id, start_date, n_days=365, project_lifetime=20, wacc=0.1, tax=0):
        self.user_id=user_id
        self.project_id=project_id
        self.start_datetime = pd.to_datetime(start_date).to_pydatetime()
        self.dt_index = pd.date_range(self.start_datetime,
                                      self.start_datetime + pd.to_timedelta(n_days, unit="D"),
                                      freq='H',
                                      closed='left')
        self.n_days = n_days
        self.project_lifetime = project_lifetime
        self.wacc = wacc
        self.tax = tax
        self.crf = (self.wacc * (1 + self.wacc) ** self.project_lifetime) / ((1 + self.wacc) ** self.project_lifetime - 1)

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
        capex = first_time_investment
        for count_of_replacements in range(1, number_of_investments):
            if count_of_replacements * component_lifetime != self.project_lifetime:
                    capex += first_time_investment / (
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
            ) / ((1 + self.wacc) ** self.project_lifetime)
        return capex
