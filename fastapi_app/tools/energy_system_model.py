from __future__ import division
import numpy as np
import pandas as pd
import oemof.solph as solph
from datetime import datetime, timedelta
import pyomo.environ as po
from fastapi_app.tools.general_optimizer_obj import Optimizer
from fastapi_app.db import sa_tables

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
        if pv["settings"]["is_selected"] is True or battery["settings"]["is_selected"] is True:
            inverter["settings"]["is_selected"] = True
        if diesel_genset["settings"]["is_selected"] is False:
            inverter["settings"]["is_selected"] = True
            battery["settings"]["is_selected"] = True
            pv["settings"]["is_selected"] = True
        self.solver = solver
        self.pv = pv
        self.diesel_genset = diesel_genset
        self.battery = battery
        self.inverter = inverter
        self.rectifier = rectifier
        self.shortage = shortage
        self.solar_potential = solar_potential
        self.demand = demand
        self.infeasible = False


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
                            investment=solph.Investment(ep_costs=self.epc["diesel_genset"] * self.n_days / 365,),
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
                            variable_costs=self.diesel_genset["parameters"]["variable_cost"],
                            investment=solph.Investment(ep_costs=self.epc["diesel_genset"] * self.n_days / 365
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
                        nominal_value=self.diesel_genset["parameters"]["nominal_capacity"],
                        variable_costs=self.diesel_genset["parameters"]["variable_cost"],
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
                ),
                inputs={b_el_dc: solph.Flow(variable_costs=0)},
                outputs={b_el_dc: solph.Flow(investment=solph.Investment(ep_costs=0))},
                initial_storage_level=self.battery["parameters"]["soc_max"],
                min_storage_level=self.battery["parameters"]["soc_min"],
                max_storage_level=self.battery["parameters"]["soc_max"],
                balanced=False,
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
                initial_storage_level=self.battery["parameters"]["soc_max"],
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

        res = model.solve(solver=self.solver,
            solve_kwargs={"tee": True},
            cmdline_options=solver_option[self.solver],)
        self.model = model
        if model.solutions.__len__() > 0:
            energy_system.results["meta"] = solph.processing.meta_results(model)
            self.results_main = solph.processing.results(model)

            self.process_results()
        else:
            print("No solution found")
        if list(res['Solver'])[0]['Termination condition'] == 'infeasible':
            self.infeasible = True

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

        self.sequences_fuel_consumption_kWh \
            = (results_fuel_source["sequences"][(("fuel_source", "fuel"), "flow")])  # conversion: kWh

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


def get_emissions(ensys_opt, user_id, project_id,):
    if ensys_opt.capacity_genset < 60:
        co2_emission_factor = 1.580
    elif ensys_opt.capacity_genset < 300:
        co2_emission_factor = 0.883
    else:
        co2_emission_factor = 0.699
    # store fuel co2 emissions (kg_CO2 per L of fuel)
    df = pd.DataFrame()
    df["non_renewable_electricity_production"] = (
                np.cumsum(ensys_opt.demand) * co2_emission_factor / 1000)  # tCO2 per year
    df["hybrid_electricity_production"] = np.cumsum(
        ensys_opt.sequences_genset) * co2_emission_factor / 1000  # tCO2 per year
    df["co2_savings"] = \
        df.loc[:, "non_renewable_electricity_production"] - df.loc[:, "hybrid_electricity_production"]  # tCO2 per year
    df['h'] = np.arange(1, len(ensys_opt.demand) + 1)
    df = df.round(3)
    emissions = models.Emissions()
    emissions.id = user_id
    emissions.project_id = project_id
    emissions.data = df.reset_index(drop=True).to_json()
    return df, emissions, co2_emission_factor


def calculate_annualized_cost(value, n_days):
    return value / n_days * 365


def get_results_df(ensys_opt,
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
                    co2_emission_factor):
    df.loc[0, "cost_renewable_assets"] = ensys_opt.total_renewable / n_days * 365
    df.loc[0, "cost_non_renewable_assets"] = ensys_opt.total_non_renewable / n_days * 365
    df.loc[0, "cost_fuel"] = ensys_opt.total_fuel / n_days * 365
    df.loc[0, "epc_total"] = (ensys_opt.total_revenue + df.loc[0, "cost_grid"]) / n_days * 365
    df.loc[0, "lcoe"] = (100 * (ensys_opt.total_revenue + df.loc[0, "cost_grid"]) / ensys_opt.total_demand)
    df.loc[0, "cost_grid"] = df.loc[0, "cost_grid"] / n_days * 365
    df.loc[0, "res"] = ensys_opt.res
    df.loc[0, "shortage_total"] = ensys_opt.shortage
    df.loc[0, "surplus_rate"] = ensys_opt.surplus_rate
    df.loc[0, "pv_capacity"] = ensys_opt.capacity_pv
    df.loc[0, "battery_capacity"] = ensys_opt.capacity_battery
    df.loc[0, "inverter_capacity"] = ensys_opt.capacity_inverter
    df.loc[0, "rectifier_capacity"] = ensys_opt.capacity_rectifier
    df.loc[0, "diesel_genset_capacity"] = ensys_opt.capacity_genset
    df.loc[0, "peak_demand"] = ensys_opt.demand.max()
    df.loc[0, "surplus"] = ensys_opt.sequences_surplus.max()
    df.loc[0, "infeasible"] = ensys_opt.infeasible
    # data for sankey diagram - all in MWh
    df.loc[0, "fuel_to_diesel_genset"] = (ensys_opt.sequences_fuel_consumption.sum() * 0.846 *
                                          ensys_opt.diesel_genset["parameters"]["fuel_lhv"] / 1000)
    df.loc[0, "diesel_genset_to_rectifier"] = (ensys_opt.sequences_rectifier.sum() /
                                               ensys_opt.rectifier["parameters"]["efficiency"] / 1000)
    df.loc[0, "diesel_genset_to_demand"] = (ensys_opt.sequences_genset.sum() / 1000
                                            - df.loc[0, "diesel_genset_to_rectifier"])
    df.loc[0, "rectifier_to_dc_bus"] = ensys_opt.sequences_rectifier.sum() / 1000
    df.loc[0, "pv_to_dc_bus"] = ensys_opt.sequences_pv.sum() / 1000
    df.loc[0, "battery_to_dc_bus"] = ensys_opt.sequences_battery_discharge.sum() / 1000
    df.loc[0, "dc_bus_to_battery"] = ensys_opt.sequences_battery_charge.sum() / 1000
    if ensys_opt.inverter["parameters"]["efficiency"] > 0:
        div = ensys_opt.inverter["parameters"]["efficiency"]
    else:
        div = 1
    df.loc[0, "dc_bus_to_inverter"] = (ensys_opt.sequences_inverter.sum() /
                                       div / 1000)
    df.loc[0, "dc_bus_to_surplus"] = ensys_opt.sequences_surplus.sum() / 1000
    df.loc[0, "inverter_to_demand"] = ensys_opt.sequences_inverter.sum() / 1000
    df.loc[0, "time_energy_system_design"] = end_execution_time - start_execution_time
    df.loc[0, "co2_savings"] = co2_savings / n_days * 365
    df.loc[0, "total_annual_consumption"] = demand_full_year.iloc[:, 0].sum()
    df.loc[0, "average_annual_demand_per_consumer"] = demand_full_year.iloc[:, 0].mean() / num_households * 1000
    df.loc[0, "base_load"] = demand_full_year.iloc[:, 0].quantile(0.1)
    df.loc[0, "max_shortage"] = (ensys_opt.sequences_shortage / ensys_opt.demand).max() * 100
    n_poles = nodes[nodes['node_type'] == 'pole'].__len__()

    links = pd.read_json(links.data) if links is not None else pd.DataFrame()
    length_dist_cable = links[links['link_type'] == 'distribution']['length'].sum()
    length_conn_cable = links[links['link_type'] == 'connection']['length'].sum()

    df.loc[0, "upfront_invest_grid"] \
        = n_poles * grid_input_parameter.loc[0, "pole_capex"] + \
          length_dist_cable * grid_input_parameter.loc[0, "distribution_cable_capex"] + \
          length_conn_cable * grid_input_parameter.loc[0, "connection_cable_capex"] + \
          num_households * grid_input_parameter.loc[0, "mg_connection_cost"]
    df.loc[0, "upfront_invest_diesel_gen"] = df.loc[0, "diesel_genset_capacity"] \
                                             * energy_system_design['diesel_genset']['parameters']['capex']
    df.loc[0, "upfront_invest_pv"] = df.loc[0, "pv_capacity"] \
                                     * energy_system_design['pv']['parameters']['capex']
    df.loc[0, "upfront_invest_inverter"] = df.loc[0, "inverter_capacity"] \
                                           * energy_system_design['inverter']['parameters']['capex']
    df.loc[0, "upfront_invest_rectifier"] = df.loc[0, "rectifier_capacity"] \
                                            * energy_system_design['rectifier']['parameters']['capex']
    df.loc[0, "upfront_invest_battery"] = df.loc[0, "battery_capacity"] \
                                          * energy_system_design['battery']['parameters']['capex']
    df.loc[0, "co2_emissions"] = ensys_opt.sequences_genset.sum() * co2_emission_factor / 1000 / n_days * 365
    df.loc[0, "fuel_consumption"] = ensys_opt.sequences_fuel_consumption_kWh.sum() / n_days * 365
    df.loc[0, "epc_pv"] = ensys_opt.epc['pv'] * ensys_opt.capacity_pv
    df.loc[0, "epc_diesel_genset"] = (ensys_opt.epc["diesel_genset"] * ensys_opt.capacity_genset) \
                                     + ensys_opt.diesel_genset["parameters"]["variable_cost"] \
                                     * ensys_opt.sequences_genset.sum(axis=0) * 365 / n_days
    df.loc[0, "epc_inverter"] = ensys_opt.epc['inverter'] * ensys_opt.capacity_inverter
    df.loc[0, "epc_rectifier"] = ensys_opt.epc["rectifier"] * ensys_opt.capacity_rectifier
    df.loc[0, "epc_battery"] = ensys_opt.epc['battery'] * ensys_opt.capacity_battery

    df = df.astype(float).round(3)
    return df

def get_energy_flow(ensys_opt, user_id, project_id):
    energy_flow_df = pd.DataFrame({
        "diesel_genset_production": ensys_opt.sequences_genset,
        "pv_production": ensys_opt.sequences_pv,
        "battery_charge": ensys_opt.sequences_battery_charge,
        "battery_discharge": ensys_opt.sequences_battery_discharge,
        "battery_content": ensys_opt.sequences_battery_content,
        "demand": ensys_opt.sequences_demand,
        "surplus": ensys_opt.sequences_surplus
    }).round(3)

    energy_flow = models.EnergyFlow()
    energy_flow.id = user_id
    energy_flow.project_id = project_id
    energy_flow.data = energy_flow_df.reset_index(drop=True).to_json()
    return energy_flow

def get_demand_coverage(ensys_opt, user_id, project_id):
    df = pd.DataFrame()
    df["demand"] = ensys_opt.sequences_demand
    df["renewable"] = ensys_opt.sequences_inverter
    df["non_renewable"] = ensys_opt.sequences_genset
    df["surplus"] = ensys_opt.sequences_surplus
    df.index.name = "dt"
    df = df.reset_index()
    df = df.round(3)
    demand_coverage = models.DemandCoverage()
    demand_coverage.id = user_id
    demand_coverage.project_id = project_id
    demand_coverage.data = df.reset_index(drop=True).to_json()
    return demand_coverage


def get_demand_curve(ensys_opt, user_id, project_id):
    df = pd.DataFrame()
    df["diesel_genset_percentage"] = (100 * np.arange(1, len(ensys_opt.sequences_genset) + 1)
                                      / len(ensys_opt.sequences_genset))
    df["diesel_genset_duration"] = (100 * np.sort(ensys_opt.sequences_genset)[::-1]/ ensys_opt.sequences_genset.max())
    df["pv_percentage"] = (100 * np.arange(1, len(ensys_opt.sequences_pv) + 1) / len(ensys_opt.sequences_pv))
    if ensys_opt.sequences_pv.max() > 0:
        div = ensys_opt.sequences_pv.max()
    else:
        div = 1
    df["pv_duration"] = (100 * np.sort(ensys_opt.sequences_pv)[::-1] / div)
    df["rectifier_percentage"] = (100 * np.arange(1, len(ensys_opt.sequences_rectifier) + 1)
                                  / len(ensys_opt.sequences_rectifier))
    if not ensys_opt.sequences_rectifier.abs().sum() == 0:
        df["rectifier_duration"] = 100 * np.nan_to_num(np.sort(ensys_opt.sequences_rectifier)[::-1]
                                                       / ensys_opt.sequences_rectifier.max())
    else:
        df["rectifier_duration"] = 0
    df["inverter_percentage"] = (100 * np.arange(1, len(ensys_opt.sequences_inverter) + 1)
                                 / len(ensys_opt.sequences_inverter))
    if ensys_opt.sequences_inverter.max() > 0:
        div = ensys_opt.sequences_inverter.max()
    else:
        div = 1
    df["inverter_duration"] = (100 * np.sort(ensys_opt.sequences_inverter)[::-1]/ div)
    df["battery_charge_percentage"] = (100 * np.arange(1, len(ensys_opt.sequences_battery_charge) + 1)
                                       / len(ensys_opt.sequences_battery_charge))
    if not ensys_opt.sequences_battery_charge.max() > 0:
        div = 1
    else:
        div = ensys_opt.sequences_battery_charge.max()
    df["battery_charge_duration"] = (100 * np.sort(ensys_opt.sequences_battery_charge)[::-1] / div)
    df["battery_discharge_percentage"] = (100 * np.arange(1, len(ensys_opt.sequences_battery_discharge) + 1)
                                          / len(ensys_opt.sequences_battery_discharge))
    if ensys_opt.sequences_battery_discharge.max() > 0:
        div = ensys_opt.sequences_battery_discharge.max()
    else:
        div = 1
    df["battery_discharge_duration"] = (100 * np.sort(ensys_opt.sequences_battery_discharge)[::-1]/ div)
    df['h'] = np.arange(1, len(ensys_opt.sequences_genset) + 1)
    df = df.round(3)
    demand_curve = models.DurationCurve()
    demand_curve.id = user_id
    demand_curve.project_id = project_id
    demand_curve.data = df.reset_index(drop=True).to_json()
    return demand_curve