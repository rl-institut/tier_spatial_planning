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
              opacity: 0.5,
              marker: {
                line: {
                  color: 'rgb(8,48,107)',
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
        showlegend: false
        }
      
      Plotly.newPlot(lcoeBreakdown, data, layout)
  }
};

// SANKEY DIAGRAM
sankeyDiagram = document.getElementById('sankeyDiagram');
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
        value:  [376.709, 11.359, 112.955, 11.132, 93.899, 22.014, 22.177, 98.162, 1.706, 96.198],
        label:  ['Fuel supplied to the diesel genset', 
                'Diesel genset output sent to the rectifier',
                'AC demand covered by the diesel genset',
                'Diesel genset electricity converted to DC',
                'PV electricity generation',
                'Battery discharge',
                'Battery charge',
                'DC electricity sent to the inverter',
                'DC excess sink',
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
  
// DEMAND COVERAGE PLOT
function makeplot() {
    var xhr = new XMLHttpRequest();
    url = 'visualization_demand_coverage/';
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send()

    xhr.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            // push nodes to the map
            demand_coverage = this.response;

            var time = [], non_renewable = [], renewable = [], demand = [], excess = [];
                
            for (var i=0; i<Object.keys(demand_coverage['genset']).length; i++) {
              time.push( i );
              non_renewable.push( demand_coverage['genset'][i] );
              renewable.push( demand_coverage['demand'][i] );
              demand.push( demand_coverage['demand'][i] );
              excess.push( demand_coverage['excess'][i]);
            }
    
            var demandCoverage = document.getElementById("demandCoverage");
            var trace1 = {
              x: time,
              y: non_renewable,
              mode: 'none',
              fill: 'tozeroy',
              name: 'Non-Renewable'
              
            };
            var trace2 = {
              x: time,
              y: renewable,
              mode: 'none',
              fill: 'tonexty',
              name: 'Renewable'
              
            };
            var trace3 = {
                x: time,
                y: demand,
                mode: 'line',
                name: 'Demand'
            };
            var trace4 = {
                x: time,
                y: excess,
                mode: 'none',
                fill: 'tonexty',
                name: 'Excess',
            };
          
            var data = [trace1, trace2, trace3, trace4];
        
            Plotly.newPlot(demandCoverage, data,
              {title: 'Demand coverage by renewable and non-renewable sources of energy'});
        }
    };

};

makeplot();