import time
import os
import numpy as np
import pandas as pd
import oemof.solph as solph
import pyomo.environ as po
from fastapi_app.helper.error_logger import logger as error_logger
from fastapi_app.models.base_optimizer import BaseOptimizer
from fastapi_app.db import sa_tables
from fastapi_app.db import sync_inserts, sync_queries
from fastapi_app import config
from fastapi_app.helper.mail import send_mail
from fastapi_app.inputs import solar_potential


def optimize_energy_system(user_id, project_id):
    try:
        ensys_opt = EnergySystemOptimizer(user_id=user_id, project_id=project_id)
        ensys_opt.optimize_energy_system()
        ensys_opt.results_to_db()
        return True
    except Exception as exc:
        user_name = 'user with user_id: {}'.format(user_id)
        error_logger.error_log(exc, 'no request', user_name)
        raise exc


class EnergySystemOptimizer(BaseOptimizer):
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
            user_id,
            project_id,
    ):
        """
        Initialize the grid optimizer object
        """
        print('start es opt')
        project_setup = sync_queries.get_input_df(user_id, project_id).iloc[0].to_dict()
        super().__init__(user_id,
                         project_id,
                         project_setup["start_date"],
                         min(project_setup["n_days"], int(os.environ.get('MAX_DAYS', 365))),
                         project_setup["project_lifetime"],
                         project_setup["interest_rate"] / 100,
                         tax=0)
        energy_system_design = sync_queries.get_energy_system_design(user_id, project_id)
        if energy_system_design['pv']["settings"]["is_selected"] is True or \
                energy_system_design['battery']["settings"]["is_selected"] is True:
            energy_system_design['inverter']["settings"]["is_selected"] = True
        if energy_system_design['diesel_genset']["settings"]["is_selected"] is False:
            energy_system_design['inverter']["settings"]["is_selected"] = True
            energy_system_design['battery']["settings"]["is_selected"] = True
            energy_system_design['pv']["settings"]["is_selected"] = True
        solver = 'gurobi' if po.SolverFactory('gurobi').available() else 'cbc'
        if solver == 'cbc':
            energy_system_design['diesel_genset']['settings']['offset'] = False
        self.solver = solver
        self.pv = energy_system_design['pv']
        self.diesel_genset = energy_system_design['diesel_genset']
        self.battery = energy_system_design['battery']
        self.inverter = energy_system_design['inverter']
        self.rectifier = energy_system_design['rectifier']
        self.shortage = energy_system_design['shortage']
        nodes = sync_queries.get_model_instance(sa_tables.Nodes, user_id, project_id)
        nodes = pd.read_json(nodes.data)
        if not nodes[nodes['consumer_type'] == 'power_house'].empty:
            lat, lon = nodes[nodes['consumer_type'] == 'power_house']['latitude', 'longitude'].to_list()
        else:
            lat, lon = nodes[['latitude', 'longitude']].mean().to_list()
        self.solar_potential = solar_potential.get_dc_feed_in_sync_db_query(lat, lon, self.dt_index).loc[self.dt_index]
        demand_opt_dict = sync_queries.get_model_instance(sa_tables.Demand, user_id, project_id).to_dict()
        self.demand_full_year = queries_demand.get_demand_time_series(nodes, demand_opt_dict).to_frame('Demand')
        self.demand = self.demand_full_year.loc[self.dt_index]['Demand'].copy()
        self.solar_potential_peak = self.solar_potential.max()
        self.demand_peak = self.demand.max()
        self.infeasible = False
        self.num_households = len(nodes[(nodes['consumer_type'] == 'household') &
                                   (nodes['is_connected'] == True)].index)
        self.nodes = nodes
        links = sync_queries.get_model_instance(sa_tables.Links, user_id, project_id)
        self.links = pd.read_json(links.data)
        self.energy_system_design = energy_system_design
        self.project_setup = project_setup


    def optimize_energy_system(self):
        # define an empty dictionary for all epc values
        start_execution_time = time.monotonic()
        self.epc = {}
        energy_system = solph.EnergySystem(timeindex=self.dt_index.copy(), infer_last_interval=True)
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
        if self.pv["settings"]["is_selected"] is False:
            pv = solph.components.Source(
                label="pv",
                outputs={b_el_dc: solph.Flow(nominal_value=0)},
            )
        elif self.pv["settings"]["design"] is True:
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
        if self.diesel_genset["settings"]["is_selected"] is False:
            diesel_genset = solph.components.Transformer(
                label="diesel_genset",
                inputs={b_fuel: solph.Flow()},
                outputs={b_el_ac: solph.Flow(nominal_value=0)},
            )
        elif self.diesel_genset["settings"]["design"] is True:
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

        if self.rectifier["settings"]["is_selected"] is False:
            rectifier = solph.components.Transformer(
                label="rectifier",
                inputs={b_el_ac: solph.Flow(nominal_value=0)},
                outputs={b_el_dc: solph.Flow()},
            )
        elif self.rectifier["settings"]["design"] is True:
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
        if self.inverter["settings"]["is_selected"] is False:
            inverter = solph.components.Transformer(
                label="inverter",
                inputs={b_el_dc: solph.Flow(nominal_value=0)},
                outputs={b_el_ac: solph.Flow()},
            )
        elif self.inverter["settings"]["design"] is True:
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

        if self.battery["settings"]["is_selected"] is False:
            battery = solph.components.GenericStorage(
                label="battery",
                nominal_storage_capacity=0,
                inputs={b_el_dc: solph.Flow()},
                outputs={b_el_dc: solph.Flow()},
            )
        elif self.battery["settings"]["design"] is True:
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
        self.execution_time = time.monotonic() - start_execution_time

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

            self._process_results()
        else:
            print("No solution found")
        if list(res['Solver'])[0]['Termination condition'] == 'infeasible':
            self.infeasible = True

    def _process_results(self):
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
        if self.diesel_genset["settings"]["is_selected"] is False:
            self.capacity_genset = 0
        elif self.diesel_genset["settings"]["design"] is True:
            self.capacity_genset = results_diesel_genset["scalars"][
                (("diesel_genset", "electricity_ac"), "invest")
            ]
        else:
            self.capacity_genset = self.diesel_genset["parameters"]["nominal_capacity"]

        if self.pv["settings"]["is_selected"] is False:
            self.capacity_pv = 0
        elif self.pv["settings"]["design"] is True:
            self.capacity_pv = results_pv["scalars"][
                (("pv", "electricity_dc"), "invest")
            ]
        else:
            self.capacity_pv = self.pv["parameters"]["nominal_capacity"]

        if self.inverter["settings"]["is_selected"] is False:
            self.capacity_inverter = 0
        elif self.inverter["settings"]["design"] is True:
            self.capacity_inverter = results_inverter["scalars"][
                (("electricity_dc", "inverter"), "invest")
            ]
        else:
            self.capacity_inverter = self.inverter["parameters"]["nominal_capacity"]

        if self.rectifier["settings"]["is_selected"] is False:
            self.capacity_rectifier = 0
        elif self.rectifier["settings"]["design"] is True:
            self.capacity_rectifier = results_rectifier["scalars"][
                (("electricity_ac", "rectifier"), "invest")
            ]
        else:
            self.capacity_rectifier = self.rectifier["parameters"]["nominal_capacity"]

        if self.battery["settings"]["is_selected"] is False:
            self.capacity_battery = 0
        elif self.battery["settings"]["design"] is True:
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

    def results_to_db(self):
        if self.model.solutions.__len__() == 0:
            if self.infeasible is True:
                df = sync_queries.get_df(sa_tables.Results, self.user_id, self.project_id)
                df.loc[0, "infeasible"] = self.infeasible
                sync_inserts.insert_results_df(df, self.user_id, self.project_id)
            return False
        self._emissions_to_db()
        self._results_to_db()
        self._energy_flow_to_db()
        self._demand_curve_to_db()
        self._project_setup_to_db()
        return True

    def _project_setup_to_db(self):
        project_setup = sync_queries.get_model_instance(sa_tables.ProjectSetup, self.user_id, self.project_id)
        project_setup.status = "finished"
        if project_setup.email_notification is True:
            user = sync_queries.get_user_by_id(self.user_id)
            subject = "PeopleSun: Model Calculation finished"
            msg = "The calculation of your optimization model is finished. You can view the results at: " \
                  "\n\n{}/simulation_results?project_id={}\n".format(config.DOMAIN, self.project_id)
            send_mail(user.email, msg, subject=subject)
        project_setup.email_notification = False
        sync_inserts.merge_model(project_setup)

    def _get_demand_coverage(self):
        df = pd.DataFrame()
        df["demand"] = self.sequences_demand
        df["renewable"] = self.sequences_inverter
        df["non_renewable"] = self.sequences_genset
        df["surplus"] = self.sequences_surplus
        df.index.name = "dt"
        df = df.reset_index()
        df = df.round(3)
        demand_coverage = sa_tables.DemandCoverage()
        demand_coverage.id = self.user_id
        demand_coverage.project_id = self.project_id
        demand_coverage.data = df.reset_index(drop=True).to_json()
        return demand_coverage


    def _emissions_to_db(self):
        if self.capacity_genset < 60:
            co2_emission_factor = 1.580
        elif self.capacity_genset < 300:
            co2_emission_factor = 0.883
        else:
            co2_emission_factor = 0.699
        # store fuel co2 emissions (kg_CO2 per L of fuel)
        df = pd.DataFrame()
        df["non_renewable_electricity_production"] = (
                np.cumsum(self.demand) * co2_emission_factor / 1000)  # tCO2 per year
        df["hybrid_electricity_production"] = np.cumsum(
            self.sequences_genset) * co2_emission_factor / 1000  # tCO2 per year
        df["co2_savings"] = \
            df.loc[:, "non_renewable_electricity_production"] - df.loc[:, "hybrid_electricity_production"]  # tCO2 per year
        df['h'] = np.arange(1, len(self.demand) + 1)
        df = df.round(3)
        emissions = sa_tables.Emissions()
        emissions.id = self.user_id
        emissions.project_id = self.project_id
        emissions.data = df.reset_index(drop=True).to_json()
        self.co2_savings = df["co2_savings"].max()
        self.co2_emission_factor = co2_emission_factor
        sync_inserts.merge_model(emissions)

    def _energy_flow_to_db(self):
        energy_flow_df = pd.DataFrame({
            "diesel_genset_production": self.sequences_genset,
            "pv_production": self.sequences_pv,
            "battery_charge": self.sequences_battery_charge,
            "battery_discharge": self.sequences_battery_discharge,
            "battery_content": self.sequences_battery_content,
            "demand": self.sequences_demand,
            "surplus": self.sequences_surplus}).round(3)
        energy_flow = sa_tables.EnergyFlow()
        energy_flow.id = self.user_id
        energy_flow.project_id = self.project_id
        energy_flow.data = energy_flow_df.reset_index(drop=True).to_json()
        sync_inserts.merge_model(energy_flow)

    def _demand_curve_to_db(self):
        df = pd.DataFrame()
        df["diesel_genset_percentage"] = (100 * np.arange(1, len(self.sequences_genset) + 1)
                                          / len(self.sequences_genset))
        df["diesel_genset_duration"] = (100 * np.sort(self.sequences_genset)[::-1] / self.sequences_genset.max())
        df["pv_percentage"] = (100 * np.arange(1, len(self.sequences_pv) + 1) / len(self.sequences_pv))
        if self.sequences_pv.max() > 0:
            div = self.sequences_pv.max()
        else:
            div = 1
        df["pv_duration"] = (100 * np.sort(self.sequences_pv)[::-1] / div)
        df["rectifier_percentage"] = (100 * np.arange(1, len(self.sequences_rectifier) + 1)
                                      / len(self.sequences_rectifier))
        if not self.sequences_rectifier.abs().sum() == 0:
            df["rectifier_duration"] = 100 * np.nan_to_num(np.sort(self.sequences_rectifier)[::-1]
                                                           / self.sequences_rectifier.max())
        else:
            df["rectifier_duration"] = 0
        df["inverter_percentage"] = (100 * np.arange(1, len(self.sequences_inverter) + 1)
                                     / len(self.sequences_inverter))
        if self.sequences_inverter.max() > 0:
            div = self.sequences_inverter.max()
        else:
            div = 1
        df["inverter_duration"] = (100 * np.sort(self.sequences_inverter)[::-1] / div)
        df["battery_charge_percentage"] = (100 * np.arange(1, len(self.sequences_battery_charge) + 1)
                                           / len(self.sequences_battery_charge))
        if not self.sequences_battery_charge.max() > 0:
            div = 1
        else:
            div = self.sequences_battery_charge.max()
        df["battery_charge_duration"] = (100 * np.sort(self.sequences_battery_charge)[::-1] / div)
        df["battery_discharge_percentage"] = (100 * np.arange(1, len(self.sequences_battery_discharge) + 1)
                                              / len(self.sequences_battery_discharge))
        if self.sequences_battery_discharge.max() > 0:
            div = self.sequences_battery_discharge.max()
        else:
            div = 1
        df["battery_discharge_duration"] = (100 * np.sort(self.sequences_battery_discharge)[::-1] / div)
        df['h'] = np.arange(1, len(self.sequences_genset) + 1)
        df = df.round(3)
        demand_curve = sa_tables.DurationCurve()
        demand_curve.id = self.user_id
        demand_curve.project_id = self.project_id
        demand_curve.data = df.reset_index(drop=True).to_json()
        sync_inserts.merge_model(demand_curve)


    def _results_to_db(self):
        results = sync_queries.get_model_instance(sa_tables.Results, self.user_id, self.project_id)
        results.cost_renewable_assets = self.total_renewable / self.n_days * 365
        results.cost_non_renewable_assets = self.total_non_renewable / self.n_days * 365
        results.cost_fuel = self.total_fuel / self.n_days * 365
        results.epc_total = (self.total_revenue + results.cost_grid) / self.n_days * 365
        results.lcoe = (100 * (self.total_revenue + results.cost_grid) / self.total_demand)
        results.cost_grid = results.cost_grid / self.n_days * 365
        results.res = self.res
        results.shortage_total = self.shortage
        results.surplus_rate = self.surplus_rate
        results.pv_capacity = self.capacity_pv
        results.battery_capacity = self.capacity_battery
        results.inverter_capacity = self.capacity_inverter
        results.rectifier_capacity = self.capacity_rectifier
        results.diesel_genset_capacity = self.capacity_genset
        results.peak_demand = self.demand.max()
        results.surplus = self.sequences_surplus.max()
        results.infeasible = self.infeasible
        # data for sankey diagram - all in MWh
        results.fuel_to_diesel_genset = (self.sequences_fuel_consumption.sum() * 0.846 *
                                              self.diesel_genset["parameters"]["fuel_lhv"] / 1000)
        results.diesel_genset_to_rectifier = (self.sequences_rectifier.sum() /
                                                   self.rectifier["parameters"]["efficiency"] / 1000)
        results.diesel_genset_to_demand = (self.sequences_genset.sum() / 1000 - results.diesel_genset_to_rectifier)
        results.rectifier_to_dc_bus = self.sequences_rectifier.sum() / 1000
        results.pv_to_dc_bus = self.sequences_pv.sum() / 1000
        results.battery_to_dc_bus = self.sequences_battery_discharge.sum() / 1000
        results.dc_bus_to_battery = self.sequences_battery_charge.sum() / 1000
        if self.inverter["parameters"]["efficiency"] > 0:
            div = self.inverter["parameters"]["efficiency"]
        else:
            div = 1
        results.dc_bus_to_inverter = (self.sequences_inverter.sum() / div / 1000)
        results.dc_bus_to_surplus = self.sequences_surplus.sum() / 1000
        results.inverter_to_demand = self.sequences_inverter.sum() / 1000
        results.time_energy_system_design = self.execution_time
        results.co2_savings = self.co2_savings / self.n_days * 365
        results.total_annual_consumption = self.demand_full_year.iloc[:, 0].sum()
        results.average_annual_demand_per_consumer = self.demand_full_year.iloc[:, 0].mean() / self.num_households * 1000
        results.base_load = self.demand_full_year.iloc[:, 0].quantile(0.1)
        results.max_shortage = (self.sequences_shortage / self.demand).max() * 100
        n_poles = self.nodes[self.nodes['node_type'] == 'pole'].__len__()
        length_dist_cable = self.links[self.links['link_type'] == 'distribution']['length'].sum()
        length_conn_cable = self.links[self.links['link_type'] == 'connection']['length'].sum()
        results.upfront_invest_grid \
            = n_poles * self.project_setup["pole_capex"] + \
              length_dist_cable * self.project_setup["distribution_cable_capex"] + \
              length_conn_cable * self.project_setup["connection_cable_capex"] + \
              self.num_households * self.project_setup["mg_connection_cost"]
        results.upfront_invest_diesel_gen = results.diesel_genset_capacity \
                                                 * self.energy_system_design['diesel_genset']['parameters']['capex']
        results.upfront_invest_pv = results.pv_capacity \
                                         * self.energy_system_design['pv']['parameters']['capex']
        results.upfront_invest_inverter = results.inverter_capacity \
                                               * self.energy_system_design['inverter']['parameters']['capex']
        results.upfront_invest_rectifier = results.rectifier_capacity \
                                                * self.energy_system_design['rectifier']['parameters']['capex']
        results.upfront_invest_battery = results.battery_capacity \
                                              * self.energy_system_design['battery']['parameters']['capex']
        results.co2_emissions = self.sequences_genset.sum() * self.co2_emission_factor / 1000 / self.n_days * 365
        results.fuel_consumption = self.sequences_fuel_consumption.sum() / self.n_days * 365
        results.epc_pv = self.epc['pv'] * self.capacity_pv
        results.epc_diesel_genset = (self.epc["diesel_genset"] * self.capacity_genset) \
                                         + self.diesel_genset["parameters"]["variable_cost"] \
                                         * self.sequences_genset.sum(axis=0) * 365 / self.n_days
        results.epc_inverter = self.epc['inverter'] * self.capacity_inverter
        results.epc_rectifier = self.epc["rectifier"] * self.capacity_rectifier
        results.epc_battery = self.epc['battery'] * self.capacity_battery
        sync_inserts.merge_model(results)
