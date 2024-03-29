// BAR DIAGRAM FOR OPTIMAL CAPACITY OF COMPONENTS
// get optimal capacities from energy system optimizer
var yValue = [0, 0, 0, 0, 0, 0, 0];
var xhr = new XMLHttpRequest();
url = "get_optimal_capacities/";
xhr.open("GET", url, true);
xhr.responseType = "json";
xhr.send();
xhr.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
        optimal_capacities = this.response;
        yValue[0] = Number(optimal_capacities['pv']);
        yValue[1] = Number(optimal_capacities['battery']);
        yValue[2] = Number(optimal_capacities['inverter']);
        yValue[3] = Number(optimal_capacities['rectifier']);
        yValue[4] = Number(optimal_capacities['diesel_genset']);
        yValue[5] = Number(optimal_capacities['peak_demand']);
        yValue[6] = Number(optimal_capacities['surplus']);
        optimalSizes = document.getElementById('optimalSizes');

        var xValue = ['PV', 'Battery', 'Inverter', 'Rectifier', 'Diesel Genset', 'Peak Demand', 'Surplus'];
        // var yValue = [61, 71, 36, 10, 65, 100, 29];
      
        var data = [
            {
              x: xValue,
              y: yValue,  
              type: 'bar',
              text: yValue.map(String),
              textposition: 'auto',
              hoverinfo: 'none',
              opacity: 0.7,
              marker: {
                color: ['rgb(8,48,107)', 'rgb(8,48,107)', 'rgb(8,48,107)', 'rgb(8,48,107)', 'rgb(8,48,107)', 'rgb(236,154,41)', 'rgb(236,154,41)'],
                line: {
                  color:'black',
                  width: 1.5
                }
              }    
            }
        ];
      
        var layout = {
            xaxis: {tickfont: {
                size: 14,
              },
              tickangle: -30,
            },
            yaxis: {
              title: 'Capacity in [kW] or [kWh]',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              }
            },
            barmode: 'stack',
            bargap: 0.5,
          };
      
        Plotly.newPlot(optimalSizes, data, layout);
      }
};

// PIE DIAGRAM FOR BREAKDOWN OF LCOE
lcoeBreakdown = document.getElementById('lcoeBreakdown');
var xhr = new XMLHttpRequest();
url = "get_lcoe_breakdown/";
xhr.open("GET", url, true);
xhr.responseType = "json";
xhr.send();
xhr.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      lcoe_breakdown = this.response;
      cost_renewable_assets = Number(lcoe_breakdown['renewable_assets']);
      cost_non_renewable_assets = Number(lcoe_breakdown['non_renewable_assets']);
      cost_grid = Number(lcoe_breakdown['grid']);
      cost_fuel = Number(lcoe_breakdown['fuel']);

      var data = [{
        type: 'pie',
        values: [cost_renewable_assets, cost_non_renewable_assets, cost_grid, cost_fuel],
        labels: ['Renewable Assets', 'Non-Renewable Assets', 'Grid', 'Fuel'],
        marker: {
            colors: ['rgb(9, 188, 138)', 'rgb(73, 89, 101)', 'rgb(236, 154, 41)', 'rgb(154, 3, 30)'],
            line: {
                color: 'black',
                width: 1.5
              }  
        },
        textinfo: 'label+percent',
        textposition: 'outside',
        automargin: true,
        opacity: 0.9,
      }]
      
      var layout = {
        // height: 400,
        // width: 400,
        margin: {'t': 0, 'b': 0, 'l': 0, 'r': 0},
        showlegend: false,
        font: {
          size: 16,
          color: 'black'
        }
      }
      Plotly.newPlot(lcoeBreakdown, data, layout)
  }
};

// SANKEY DIAGRAM
sankeyDiagram = document.getElementById('sankeyDiagram');
var xhr = new XMLHttpRequest();
url = "get_data_for_sankey_diagram/";
xhr.open("GET", url, true);
xhr.responseType = "json";
xhr.send();
xhr.onreadystatechange = function () {
  if (this.readyState == 4 && this.status == 200) {
    sankey_data = this.response;
    fuel_to_diesel_genset = Number(sankey_data['fuel_to_diesel_genset'])
    diesel_genset_to_rectifier = Number(sankey_data['diesel_genset_to_rectifier'])
    diesel_genset_to_demand = Number(sankey_data['diesel_genset_to_demand'])
    rectifier_to_dc_bus = Number(sankey_data['rectifier_to_dc_bus'])
    pv_to_dc_bus = Number(sankey_data['pv_to_dc_bus'])
    battery_to_dc_bus = Number(sankey_data['battery_to_dc_bus'])
    dc_bus_to_battery = Number(sankey_data['dc_bus_to_battery'])
    dc_bus_to_inverter = Number(sankey_data['dc_bus_to_inverter'])
    dc_bus_to_surplus = Number(sankey_data['dc_bus_to_surplus'])
    inverter_to_demand = Number(sankey_data['inverter_to_demand'])

    var data = [{
        type: 'sankey',
        orientation: 'h',
        node: {
          pad: 10,
          thickness: 20,
          valueformat: ".3f",
          valuesuffix: "MWh",
          line: {
            color: 'black',
            width: 0.5
          },
        label: ['Fuel',
                'Diesel Genset', 
                'Rectifier', 
                'PV', 
                'DC Bus', 
                'Battery',
                'Inverter',
                'Demand',
                'Surplus'],
        color: 'rgb(23, 64, 92)',
            },
      
        link: {
            source: [0, 1, 1, 2, 3, 5, 4, 4, 4, 6],
            target: [1, 2, 7, 4, 4, 4, 5, 6, 8, 7],
            value:  [fuel_to_diesel_genset, 
                     diesel_genset_to_rectifier, 
                     diesel_genset_to_demand, 
                     rectifier_to_dc_bus, 
                     pv_to_dc_bus, 
                     battery_to_dc_bus, 
                     dc_bus_to_battery, 
                     dc_bus_to_inverter, 
                     dc_bus_to_surplus, 
                     inverter_to_demand],
            label:  ['Fuel supplied to the diesel genset', 
                    'Diesel genset output sent to the rectifier',
                    'AC demand covered by the diesel genset',
                    'Diesel genset electricity converted to DC',
                    'PV electricity generation',
                    'Battery discharge',
                    'Battery charge',
                    'DC electricity sent to the inverter',
                    'DC surplus sink',
                    'AC demand covered by the PV system'],
            color: 'rgb(168, 181, 192)', 
        }
    }]
        
    var layout = {
    font: {
        size: 16,
        color: 'black'
    }
    }
      
    Plotly.react(sankeyDiagram, data, layout)
  }
};

// ENERGY FLOWS PLOT
function makeplot_energy_flows() {
  var xhr = new XMLHttpRequest();
  url = 'get_data_for_energy_flows/';
  xhr.open("GET", url, true);
  xhr.responseType = "json";
  xhr.send()

  xhr.onreadystatechange = function () {
      if (this.readyState == 4 && this.status == 200) {
          // push nodes to the map
          energy_flows = this.response;

          var time = [], diesel_genset_production = [], pv_production = [], battery_charge = [], battery_discharge = [], battery_content = [], demand = [], surplus = [];
              
          for (var i=0; i<Object.keys(energy_flows['diesel_genset_production']).length; i++) {
            time.push( i );
            diesel_genset_production.push( energy_flows['diesel_genset_production'][i] );
            pv_production.push( energy_flows['pv_production'][i] );
            battery_charge.push( energy_flows['battery_charge'][i] );
            battery_discharge.push( energy_flows['battery_discharge'][i] );
            battery_content.push( energy_flows['battery_content'][i] );
            demand.push( energy_flows['demand'][i] );
            surplus.push( energy_flows['surplus'][i] );
          }
  
          var energyFlows = document.getElementById("energyFlows");
          var trace1 = {
            x: time,
            y: diesel_genset_production,
            mode: 'lines',
            name: 'Diesel Genset',
            line: {shape: 'vhv'},
            type: 'scatter',                      
          };
          var trace2 = {
            x: time,
            y: pv_production,
            mode: 'lines',
            name: 'PV',
            line: {shape: 'vhv'},
            type: 'scatter',          
          };
          var trace3 = {
              x: time,
              y: battery_charge,
              mode: 'lines',
              name: 'Battery - Charge',
              line: {shape: 'vhv'},
              type: 'scatter',            
          };
          var trace4 = {
              x: time,
              y: battery_discharge,
              mode: 'lines',
              name: 'Battery - Discharge',
              line: {shape: 'vhv'},
              type: 'scatter',        
          };
          var trace5 = {
              x: time,
              y: battery_content,
              mode: 'lines',
              name: 'Battery - Content',
              line: {shape: 'vhv'},
              type: 'scatter',       
          };
          var trace6 = {
              x: time,
              y: demand,
              mode: 'lines',
              name: 'Demand',
              line: {shape: 'vhv'},
              type: 'scatter',            
          };
          var trace7 = {
              x: time,
              y: surplus,
              mode: 'lines',
              name: 'Surplus',
              line: {shape: 'vhv'},
              type: 'scatter',
          };
        
          var data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7];

          var layout = {
            xaxis: {
              title: 'Time in [Hour]',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              },
            },
            yaxis: {
              title: 'Energy Flow in [kW]',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              }
            },
            // title: 'Energy flows in different components of the system.',
          };
          Plotly.newPlot(energyFlows, data, layout);
      }
  };

};
makeplot_energy_flows();

// DEMAND COVERAGE PLOT
function makeplot_demand_coverage() {
  var xhr = new XMLHttpRequest();
  url = 'get_demand_coverage_data/';
  xhr.open("GET", url, true);
  xhr.responseType = "json";
  xhr.send()

  xhr.onreadystatechange = function () {
      if (this.readyState == 4 && this.status == 200) {
          // push nodes to the map
          demand_coverage = this.response;

          var time = [], renewable = [], non_renewable = [], demand = [], surplus = [];
              
          for (var i=0; i<Object.keys(demand_coverage['demand']).length; i++) {
            time.push( i );
            demand.push( demand_coverage['demand'][i] );
            renewable.push( demand_coverage['renewable'][i] );
            non_renewable.push( demand_coverage['non_renewable'][i] );
            surplus.push( demand_coverage['surplus'][i]);
          }
  
          var demandCoverage = document.getElementById("demandCoverage");
          var trace1 = {
            x: time,
            y: non_renewable,
            // mode: 'none',
            // fill: 'tozeroy',
            stackgroup: 'one',
            name: 'Non-Renewable',
          };
          var trace2 = {
            x: time,
            y: renewable,
            // mode: 'none',
            // fill: 'tonexty',
            stackgroup: 'one',
            name: 'Renewable'
            
          };
          var trace3 = {
              x: time,
              y: demand,
              mode: 'line',
              name: 'Demand',
              line: {
                    color: 'black',
                    width: 2.5
              },
            };
          var trace4 = {
              x: time,
              y: surplus,
              // mode: 'none',
              // fill: 'tonexty',
              stackgroup: 'one',
              name: 'surplus',
          };
        
          var layout = {
            xaxis: {
              title: 'Time in [Hour]',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              },
            },
            yaxis: {
              title: 'Demand in [kW]',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              }
            },
          };

          var data = [trace1, trace2, trace3, trace4];
      
          Plotly.newPlot(demandCoverage, data, layout);
      }
  };

};
makeplot_demand_coverage();


// DURATION CURVES
function makeplot_duration_curves() {
  var xhr = new XMLHttpRequest();
  url = 'get_data_for_duration_curves/';
  xhr.open("GET", url, true);
  xhr.responseType = "json";
  xhr.send()

  xhr.onreadystatechange = function () {
      if (this.readyState == 4 && this.status == 200) {
          // push nodes to the map
          duration_curves = this.response;

          var diesel_genset_percentage = [], diesel_genset_duration = [], pv_percentage = [], pv_duration = [], rectifier_percentage = [], rectifier_duration = [], inverter_percentage = [], inverter_duration = [], battery_charge_percentage = [], battery_charge_duration = [], battery_discharge_percentage = [], battery_discharge_duration = [];
              
          for (var i=0; i<Object.keys(duration_curves['diesel_genset_percentage']).length; i++) {
            diesel_genset_percentage.push( duration_curves['diesel_genset_percentage'][i] );
            diesel_genset_duration.push( duration_curves['diesel_genset_duration'][i] );
            pv_percentage.push( duration_curves['pv_percentage'][i] );
            pv_duration.push( duration_curves['pv_duration'][i] );
            rectifier_percentage.push( duration_curves['rectifier_percentage'][i] );
            rectifier_duration.push( duration_curves['rectifier_duration'][i] );
            inverter_percentage.push( duration_curves['inverter_percentage'][i] );
            inverter_duration.push( duration_curves['inverter_duration'][i] );
            battery_charge_percentage.push( duration_curves['battery_charge_percentage'][i] );
            battery_charge_duration.push( duration_curves['battery_charge_duration'][i] );
            battery_discharge_percentage.push( duration_curves['battery_discharge_percentage'][i] );
            battery_discharge_duration.push( duration_curves['battery_discharge_duration'][i] );
          }
  
          var durationCurves = document.getElementById("durationCurves");
          var trace1 = {
            x: diesel_genset_percentage,
            y: diesel_genset_duration,
            mode: 'lines',
            name: 'Diesel Genset'
            
          };
          var trace2 = {
            x: pv_percentage,
            y: pv_duration,
            mode: 'lines',
            name: 'PV'
            
          };
          var trace3 = {
            x: rectifier_percentage,
            y: rectifier_duration,
            mode: 'lines',
            name: 'Rectifier'
            
          };
          var trace4 = {
            x: inverter_percentage,
            y: inverter_duration,
            mode: 'lines',
            name: 'Inverter'
            
          };
          var trace5 = {
            x: battery_charge_percentage,
            y: battery_charge_duration,
            mode: 'lines',
            name: 'Battery - Charging'
            
          };
          var trace6 = {
            x: battery_discharge_percentage,
            y: battery_discharge_duration,
            mode: 'lines',
            name: 'Battery - Discharging'
            
          };
        
          var data = [trace1, trace2, trace3, trace4, trace5, trace6];

          var layout = {
            xaxis: {
              title: 'Percentage of Operation in [%]',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              },
            },
            yaxis: {
              title: 'Load in [%]',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              }
            },
          };
          Plotly.newPlot(durationCurves, data, layout);
      }
  };

};
makeplot_duration_curves();

// DEMAND COVERAGE PLOT
function makeplot_co2_emissions() {
  var xhr = new XMLHttpRequest();
  url = 'get_co2_emissions_data/';
  xhr.open("GET", url, true);
  xhr.responseType = "json";
  xhr.send()

  xhr.onreadystatechange = function () {
      if (this.readyState == 4 && this.status == 200) {
          // push nodes to the map
          co2_emissions = this.response;

          var time = [], non_renewable = [], hybrid = [];
              
          for (var i=0; i<Object.keys(co2_emissions['non_renewable_electricity_production']).length; i++) {
            time.push( i );
            non_renewable.push( co2_emissions['non_renewable_electricity_production'][i] );
            hybrid.push( co2_emissions['hybrid_electricity_production'][i] );
          }
  
          var co2Emissions = document.getElementById("co2Emissions");
          var trace1 = {
            x: time,
            y: non_renewable,
            mode: 'lines',
            name: 'Non-Renewable'
            
          };
          var trace2 = {
            x: time,
            y: hybrid,
            mode: 'none',
            fill: 'tonexty',
            name: 'Savings'
            
          };
          var trace3 = {
            x: time,
            y: hybrid,
            mode: 'lines',
            name: 'Hybrid'
            
          };
        
          var data = [trace1, trace2, trace3];
      
          var layout = {
            xaxis: {
              title: 'Time in [hour]',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              },
            },
            yaxis: {
              title: 'CO2 Emissions [ton]',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              }
            },
          };
          Plotly.newPlot(co2Emissions, data, layout);
      }
  };

};
makeplot_co2_emissions();