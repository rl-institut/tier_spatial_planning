from __future__ import division
import numpy as np
import pandas as pd
from k_means_constrained import KMeansConstrained
from munkres import Munkres
from scipy.sparse.csgraph import minimum_spanning_tree
from fastapi_app.tools.grids import Grid
import oemof.solph as solph
from datetime import datetime, timedelta
import pyomo.environ as po
from pyproj import Proj
from fastapi_app.io.db import sync_queries, queries_demand, models
from fastapi_app.tools.optimizer import Optimizer

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
                balanced=True,
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
