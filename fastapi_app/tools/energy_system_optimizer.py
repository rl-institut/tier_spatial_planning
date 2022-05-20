import oemof.solph as solph
import pandas as pd
from datetime import datetime, timedelta


class EnergySystemOptimizer:
    """
    This class includes:
        - methods for optimizing the "energy system" object
        - attributes containing all default values for the optimization parameters.

    Attributes
    ----------
    ???
    """

    def __init__(self, start_date, n_days):
        """
        Initialize the grid optimizer object
        """
        self.start_date = start_date
        self.n_days = n_days

    def create_datetime_objects(self):
        """
        explanation
        """
        start_date_obj = datetime.strptime(self.start_date, '%Y-%m-%d')
        self.start_date = start_date_obj.date()
        self.start_time = start_date_obj.time()
        self.start_datetime = datetime.combine(start_date_obj.date(), start_date_obj.time())
        self.end_datetime = self.start_datetime + timedelta(days=self.n_days)

# # simulation period
# start = '2022-01-01'
# n_days = 365

    def import_data(self, path):
        data = pd.read_csv(filepath_or_buffer=path, delimiter=';')
        data.index = pd.date_range(start=self.start_datetime, periods=len(data), freq='H')

        self.solar_potential = data.SolarGen.loc[self.start_datetime:self.end_datetime]
        self.demand = data.Demand.loc[self.start_datetime:self.end_datetime]
        # peak_solar_potential = solar_potential.max()
        # peak_demand = hourly_demand.max()

        # demand = pd.read_csv(filepath_or_buffer=path_demand, delimiter=';')

        # # change the index of data to be able to select data based on the time range
        # solar_potential.index = pd.date_range(
        #     start=self.start_date, periods=len(solar_potential), freq='H')
        # demand.index = pd.date_range(start=self.start_date, periods=len(demand), freq='H')

        # # choose the range of solar potential and demand based on the selected simulation period
        # self.solar_potential = solar_potential.loc[self.start_datetime:self.end_datetime]
        self.solar_potential_peak = self.solar_potential.max()

        # self.demand = demand.loc[self.start_datetime:self.end_datetime]
        self.demand_peak = self.demand.max()

    def optimize_energy_system(self):
        date_time_index = pd.date_range(start=self.start_date, periods=self.n_days * 24, freq="H")
        energy_system = solph.EnergySystem(timeindex=date_time_index)

        # -------------------- BUSES --------------------
        # create electricity and fuel buses
        b_el_ac = solph.Bus(label="electricity_ac")
        b_el_dc = solph.Bus(label="electricity_dc")
        b_fuel = solph.Bus(label="fuel")

        # -------------------- SOURCES --------------------
        fuel_source = solph.Source(
            label="fuel_source",
            outputs={
                b_fuel: solph.Flow(variable_costs=0.619 / 11.87)
            }
        )

        pv = solph.Source(
            label="pv",
            outputs={
                b_el_dc: solph.Flow(
                    fix=self.solar_potential,
                    nominal_value=None,
                    investment=solph.Investment(
                        ep_costs=137.46 * self.n_days / 365),
                    variable_costs=0,
                )
            },
        )

        # -------------------- TRANSFORMERS --------------------
        # optimize capacity of the fuel generator
        fuel_genset = solph.Transformer(
            label="fuel_genset",
            inputs={b_fuel: solph.Flow()},
            outputs={
                b_el_ac: solph.Flow(
                    nominal_value=None,
                    variable_costs=0.023,
                    investment=solph.Investment(
                        ep_costs=157.55 * self.n_days / 365),
                )
            },
            conversion_factors={b_el_ac: 0.33},
        )

        rectifier = solph.Transformer(
            label="rectifier",
            inputs={
                b_el_ac: solph.Flow(
                    nominal_value=None,
                    investment=solph.Investment(ep_costs=73.10 * self.n_days / 365),
                    variable_costs=0,
                )
            },
            outputs={b_el_dc: solph.Flow()},
            conversion_factor={
                b_el_dc: 0.98,
            },
        )

        inverter = solph.Transformer(
            label="inverter",
            inputs={
                b_el_dc: solph.Flow(
                    nominal_value=None,
                    investment=solph.Investment(ep_costs=73.10 * self.n_days / 365),
                    variable_costs=0,
                )
            },
            outputs={b_el_ac: solph.Flow()},
            conversion_factor={
                b_el_ac: 0.98,
            },
        )
        # -------------------- STORAGE --------------------
        storage = solph.GenericStorage(
            label="storage",
            nominal_storage_capacity=None,
            investment=solph.Investment(ep_costs=91.08 * self.n_days / 365),
            inputs={b_el_dc: solph.Flow(variable_costs=0)},
            outputs={b_el_dc: solph.Flow(
                investment=solph.Investment(ep_costs=0))},
            initial_storage_capacity=0.0,
            min_storage_level=0.0,
            max_storage_level=1.0,
            balanced=True,
            inflow_conversion_factor=0.9,
            outflow_conversion_factor=0.9,
            invest_relation_input_capacity=1,
            invest_relation_output_capacity=0.5,
        )

        # -------------------- SINKS --------------------
        demand_el = solph.Sink(
            label="electricity_demand",
            inputs={
                b_el_ac: solph.Flow(
                    fix=self.demand / self.demand_peak,
                    nominal_value=self.demand_peak
                )
            },
        )

        excess_sink = solph.Sink(
            label="excess_sink",
            inputs={b_el_dc: solph.Flow()},
        )

        # add all objects to the energy system
        energy_system.add(
            pv,
            fuel_source,  # sources
            b_el_dc,
            b_el_ac,
            b_fuel,  # buses
            inverter,
            rectifier,
            fuel_genset,  # transformers
            storage,  # storage
            demand_el,
            excess_sink,  # sinks
        )

        model = solph.Model(energy_system)

        # optimize the energy system
        # gurobi --> 'MipGap': '0.01'
        # cbc --> 'ratioGap': '0.01'
        model.solve(solver="gurobi", solve_kwargs={
            "tee": True}, cmdline_options={'MipGap': '0.02'})
        energy_system.results["meta"] = solph.processing.meta_results(model)
        self.results_main = solph.processing.results(model)

    def process_results(self):
        results_pv = solph.views.node(results=self.results_main, node='pv')
        results_fuel_source = solph.views.node(
            results=self.results_main, node='fuel_source')
        results_fuel_genset = solph.views.node(
            results=self.results_main, node='fuel_genset')
        results_inverter = solph.views.node(results=self.results_main, node='inverter')
        results_rectifier = solph.views.node(results=self.results_main, node='rectifier')
        results_storage = solph.views.node(results=self.results_main, node='storage')
        results_demand_el = solph.views.node(
            results=self.results_main, node='electricity_demand')
        results_excess_sink = solph.views.node(
            results=self.results_main, node='excess_sink')

        # -------------------- SEQUENCES (DYNAMIC) --------------------
        # hourly demand profile
        sequences_demand = results_demand_el['sequences'][(
            ('electricity_ac', 'electricity_demand'), 'flow')]

        # hourly profiles for solar potential and pv production
        sequences_pv = results_pv['sequences'][(('pv', 'electricity_dc'), 'flow')]

        # hourly profiles for fuel consumption and electricity production in the fuel genset
        # the 'flow' from oemof is in kWh and must be converted to liter
        sequences_fuel_consumption = results_fuel_source['sequences'][
            (('fuel_source', 'fuel'), 'flow')] / 11.87  # convert to [l]

        sequences_genset = results_fuel_genset['sequences'][
            (('fuel_genset', 'electricity_ac'), 'flow')]

        # hourly profiles for charge, discharge, and content of the storage
        sequences_storage_charge = results_storage['sequences'][
            (('electricity_dc', 'storage'), 'flow')]

        sequences_storage_discharge = results_storage['sequences'][
            (('storage', 'electricity_dc'), 'flow')]

        sequences_storage_content = results_storage['sequences'][
            (('storage', 'None'), 'storage_content')]

        # hourly profiles for inverted electricity from dc to ac
        sequences_inverter = results_inverter['sequences'][(
            ('inverter', 'electricity_ac'), 'flow')]

        # hourly profiles for inverted electricity from ac to dc
        sequences_rectifier = results_rectifier['sequences'][(
            ('rectifier', 'electricity_dc'), 'flow')]

        # hourly profiles for excess ac and dc electricity production
        sequences_excess = results_excess_sink['sequences'][
            (('electricity_dc', 'excess_sink'), 'flow')]
        # + results_excess_sink['sequences'][
        # (('electricity_ac', 'excess_sink'), 'flow')]

        # -------------------- SCALARS (STATIC) --------------------
        capacity_genset = results_fuel_genset['scalars'][
            (('fuel_genset', 'electricity_ac'), 'invest')]
        capacity_pv = results_pv['scalars'][(('pv', 'electricity_dc'), 'invest')]
        capacity_inverter = results_inverter['scalars'][(
            ('electricity_dc', 'inverter'), 'invest')]
        capacity_rectifier = results_rectifier['scalars'][(
            ('electricity_ac', 'rectifier'), 'invest')]
        capacity_storage = results_storage['scalars'][(
            ('electricity_dc', 'storage'), 'invest')]

        total_component = (157.55 * capacity_genset + 137.46 * capacity_pv
                           + 73.10 * capacity_inverter + 91.08 * capacity_storage) * self.n_days / 365
        total_variale = 0.023 * sequences_genset.sum(axis=0)
        total_fuel = 0.619 * sequences_fuel_consumption.sum(axis=0)
        total_revenue = total_component + total_variale + total_fuel
        total_demand = sequences_demand.sum(axis=0)
        lcoe = 100 * total_revenue / total_demand

        res = 100 * sequences_pv.sum(axis=0) / \
            (sequences_genset.sum(axis=0) + sequences_pv.sum(axis=0))

        excess_rate = 100 * \
            sequences_excess.sum(
                axis=0) / (sequences_genset.sum(axis=0) + sequences_pv.sum(axis=0))
        genset_to_dc = 100 * \
            sequences_rectifier.sum(axis=0) / sequences_genset.sum(axis=0)

        print('')
        print(40 * '*')
        print(f'LCOE:\t {lcoe:.2f} cent/kWh')
        print(f'RES:\t {res:.0f}%')
        print(f'Excess:\t {excess_rate:.1f}% of the total production')
        print(f'AC--DC:\t {genset_to_dc:.1f}% of the genset production')
        print(40 * '*')
        print(f'genset:\t {capacity_genset:.0f} kW')
        print(f'pv:\t {capacity_pv:.0f} kW')
        print(f'st:\t {capacity_storage:.0f} kW')
        print(f'inv:\t {capacity_inverter:.0f} kW')
        print(f'rect:\t {capacity_rectifier:.0f} kW')
        print(f'peak:\t {sequences_demand.max():.0f} kW')
        print(f'excess:\t {sequences_excess.max():.0f} kW')
        print(40 * '*')
