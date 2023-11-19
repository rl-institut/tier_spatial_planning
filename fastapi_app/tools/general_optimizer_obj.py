from __future__ import division


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
