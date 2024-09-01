document.getElementById('downloadCSV').addEventListener('click', function () {
    window.location.href = '/download_data/' + project_id + '/csv';
});
var targetNode = document.getElementById('responseMsg');
var config = {childList: true, subtree: true, characterData: true};
var callback = function (mutationsList, observer) {
    for (let mutation of mutationsList) {
        if ((mutation.type === 'childList' || mutation.type === 'characterData') && targetNode.textContent.trim() !== '') {
            var modal = document.getElementById('msgBox');
            modal.style.display = "block";
        }
    }
};
var observer = new MutationObserver(callback);
observer.observe(targetNode, config);
document.getElementById("msgBox").style.zIndex = "9999";

// plot functions used for the plots in results page of web app.
// Functions are called from function plot in backend_communication.js


function plot_bar_chart(data) {
    let yValue = [0, 0, 0, 0, 0, 0, 0];
    let yValue2 = [0];
    let optimal_capacities = data;
    yValue[0] = Number(optimal_capacities['pv']);
    yValue[1] = Number(optimal_capacities['inverter']);
    yValue[2] = Number(optimal_capacities['rectifier']);
    yValue[3] = Number(optimal_capacities['diesel_genset']);
    yValue[4] = Number(optimal_capacities['peak_demand']);
    yValue[5] = Number(optimal_capacities['surplus']);
    yValue2 = Number(optimal_capacities['battery']);
    let optimalSizes = document.getElementById('optimalSizes');
    let xValue = ['PV  ',
        'Inverter  ',
        'Rectifier  ',
        'Diesel Genset  ',
        'Peak Demand  ',
        'Max. Surplus  ',
        'Battery  '];
    // Reverse the arrays
    xValue = xValue.reverse();
    yValue = yValue.reverse();
    let colors = ['rgb(8,48,107)', 'rgb(8,48,107)', 'rgb(8,48,107)', 'rgb(8,48,107)',
        'rgb(8,48,107)', 'rgb(8,48,107)', 'rgb(133, 52, 124)'];
    colors = colors.reverse();  // Reverse the color array
    var data = [
        {
            y: xValue,
            x: yValue,
            xaxis: 'x1',
            type: 'bar',
            orientation: 'h',
            text: yValue.map(String),
            textposition: 'auto',
            hoverinfo: 'none',
            opacity: 0.7,
            marker: {
                color: colors,
                line: {
                    color: 'black',
                    width: 1.5
                }
            },
            showlegend: false
        },
        {
            y: ['Battery  '],
            x: [yValue2],
            xaxis: 'x2',
            type: 'bar',
            orientation: 'h',
            marker: {
                color: 'rgb(133, 52, 124)'
            },
            showlegend: false
        }
    ];
    const layout = {
        plot_bgcolor: '#FAFAFA',
        paper_bgcolor: '#FAFAFA',
        yaxis: {
            tickfont: {
                size: 14,
            },
            tickangle: -30,
        },
        xaxis: {
            title: 'Capacity in kW',
            titlefont: {
                color: 'rgb(8,48,107)',
                size: 16,
            },
            tickfont: {
                color: 'rgb(8,48,107)',
                size: 14,
            },
            side: 'top'
        },
        xaxis2: {
            title: 'Capacity in kWh',
            showgrid: false,
            zeroline: false,
            titlefont: {
                color: 'rgb(133, 52, 124)',
                size: 16,
            },
            tickfont: {
                color: 'rgb(133, 52, 124)',
                size: 14,
            },
            overlaying: 'x',
            side: 'bottom'
        },
        barmode: 'stack',
        bargap: 0.5,
        showlegend: false,
        autosize: false,
        margin: {
            l: 150,  // Increase left margin
            r: 50,
            b: 100,
            t: 100,
            pad: 4
        },
    };

    Plotly.newPlot(optimalSizes, data, layout);
}


function plot_lcoe_pie(lcoe_breakdown) {
    cost_renewable_assets = Number(lcoe_breakdown['renewable_assets']);
    cost_non_renewable_assets = Number(lcoe_breakdown['non_renewable_assets']);
    cost_grid = Number(lcoe_breakdown['grid']);
    cost_fuel = Number(lcoe_breakdown['fuel']);
    let data = [{
        type: 'pie',
        hole: .6,
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

    let layout = {
        plot_bgcolor: '#FAFAFA',
        paper_bgcolor: '#FAFAFA',
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


function plot_sankey(data) {

    sankey_data = data;
    fuel_to_diesel_genset = Number(sankey_data['fuel_to_diesel_genset'])
    diesel_genset_to_rectifier = Number(sankey_data['diesel_genset_to_rectifier'])
    diesel_genset_to_demand = Number(sankey_data['diesel_genset_to_demand'])
    rectifier_to_dc_bus = Number(sankey_data['rectifier_to_dc_bus'])
    pv_to_dc_bus = Number(sankey_data['pv_to_dc_bus'])
    battery_to_dc_bus = Number(sankey_data['battery_to_dc_bus'])
    dc_bus_to_battery = Number(sankey_data['dc_bus_to_battery'])
    dc_bus_to_inverter = Number(sankey_data['dc_bus_to_inverter'])
    pv_to_surplus = 0
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
            source: [0, 1, 1, 2, 3, 5, 4, 4, 3, 6], // Modified
            target: [1, 2, 7, 4, 4, 4, 5, 6, 8, 7], // Modified
            value: [fuel_to_diesel_genset,
                diesel_genset_to_rectifier,
                diesel_genset_to_demand,
                rectifier_to_dc_bus,
                pv_to_dc_bus,
                battery_to_dc_bus,
                dc_bus_to_battery,
                dc_bus_to_inverter,
                pv_to_surplus,
                inverter_to_demand],
            label: ['Fuel supplied to the diesel genset',
                'Diesel genset output sent to the rectifier',
                'AC demand covered by the diesel genset',
                'Diesel genset electricity converted to DC',
                'PV electricity generation',
                'Battery discharge',
                'Battery charge',
                'DC electricity sent to the inverter',
                'Surplus PV electricity',
                'AC demand covered by the PV system'],
            color: 'rgb(168, 181, 192)',
        }
    }]


    const layout = {
        plot_bgcolor: '#FAFAFA',
        paper_bgcolor: '#FAFAFA',
        font: {size: 16, color: 'black'}
    };
    Plotly.react(sankeyDiagram, data, layout)
}


// ENERGY FLOWS PLOT
function plot_energy_flows(energy_flows) {



    const { diesel_genset_production, pv_production, battery, battery_content, demand, surplus } = energy_flows;
    const time = Array.from({ length: pv_production.length }, (_, i) => i);

    const energyFlows = document.getElementById("energyFlows");
    const trace1 = {
        x: time,
        y: diesel_genset_production,
        mode: 'lines',
        name: 'Diesel Genset',
        line: {shape: 'hv'},
        type: 'scatter',
    };
    const trace2 = {
        x: time,
        y: pv_production,
        mode: 'lines',
        name: 'PV',
        line: {shape: 'hv'},
        type: 'scatter',
    };
    const trace3 = {
        x: time,
        y: battery,
        mode: 'lines',
        name: 'Battery In-/Output',
        line: {shape: 'hv'},
        type: 'scatter',
    };
    const trace4 = {
        x: time,
        y: battery_content,
        mode: 'lines',
        name: 'Battery Content',
        yaxis: 'y2',  //this makes sure that the trace uses the second y-axis.
        line: {shape: 'hv'},
        type: 'scatter',
        visible: 'legendonly',

    };
    const trace5 = {
        x: time,
        y: demand,
        mode: 'lines',
        name: 'Demand',
        line: {shape: 'hv'},
        type: 'scatter',
    };
    const trace6 = {
        x: time,
        y: surplus,
        mode: 'lines',
        name: 'Surplus',
        line: {shape: 'hv'},
        type: 'scatter',
    };

    const data = [trace1, trace2, trace3, trace4, trace5, trace6];

    const layout = {
        plot_bgcolor: '#FAFAFA',
        paper_bgcolor: '#FAFAFA',
        xaxis: {
            title: 'Time in hours',
            titlefont: {
                size: 16,
            },
            tickfont: {
                size: 14,
            },
        },
        yaxis: {
            title: 'Energy Flow in kW',
            titlefont: {
                size: 16,
            },
            tickfont: {
                size: 14,
            }
        },
        yaxis2: {   // second y-axis
            title: 'Battery Content in kWh',
            overlaying: 'y',
            side: 'right',
            showgrid: false,
        },
        legend: {
            x: 1, // This positions the legend at the right edge of the chart.
            y: 1, // This positions the legend at the top of the chart.
            xanchor: 'auto', // The anchor for the x position. The 'auto' value will let Plotly decide the best location.
            yanchor: 'auto', // The anchor for the y position. The 'auto' value will let Plotly decide the best location.
            bgcolor: 'rgba(255, 255, 255, 1)', // Fully opaque white background.
            bordercolor: '#E2E2E2',
            borderwidth: 2,
        },
        autosize: true,
        // title: 'Energy flows in different components of the system.',
    };
    Plotly.newPlot(energyFlows, data, layout);
}


// DEMAND COVERAGE PLOT
function plot_demand_coverage(demand_coverage) {

    const { renewable, non_renewable, demand, surplus } = demand_coverage;
    const time = Array.from({ length: renewable.length }, (_, i) => i);


    const demandCoverage = document.getElementById("demandCoverage");
    const trace1 = {
        x: time,
        y: non_renewable,
        // mode: 'none',
        // fill: 'tozeroy',
        stackgroup: 'one',
        name: 'Non-Renewable',
    };
    const trace2 = {
        x: time,
        y: renewable,
        // mode: 'none',
        // fill: 'tonexty',
        stackgroup: 'one',
        name: 'Renewable'

    };
    const trace3 = {
        x: time,
        y: demand,
        mode: 'line',
        name: 'Demand',
        line: {
            color: 'black',
            width: 2.5
        },
    };
    const trace4 = {
        x: time,
        y: surplus,
        // mode: 'none',
        // fill: 'tonexty',
        stackgroup: 'one',
        name: 'surplus',
    };

    const layout = {
        plot_bgcolor: '#FAFAFA',
        paper_bgcolor: '#FAFAFA',
        xaxis: {
            title: 'Time in hours',
            titlefont: {
                size: 16,
            },
            tickfont: {
                size: 14,
            },
        },
        yaxis: {
            title: 'Demand in kW',
            titlefont: {
                size: 16,
            },
            tickfont: {
                size: 14,
            }
        },
    };

    const data = [trace1, trace2, trace3, trace4];

    Plotly.newPlot(demandCoverage, data, layout);
}


// DURATION CURVES
function plot_duration_curves(duration_curves) {

    const { diesel_genset_duration, pv_percentage, pv_duration, rectifier_duration, inverter_duration, battery_charge_duration,
        battery_discharge_duration } = duration_curves;


    const durationCurves = document.getElementById("durationCurves");
    const trace1 = {
        x: pv_percentage,
        y: diesel_genset_duration,
        mode: 'lines',
        name: 'Diesel Genset'

    };
    const trace2 = {
        x: pv_percentage,
        y: pv_duration,
        mode: 'lines',
        name: 'PV'

    };
    const trace3 = {
        x: pv_percentage,
        y: rectifier_duration,
        mode: 'lines',
        name: 'Rectifier'

    };
    const trace4 = {
        x: pv_percentage,
        y: inverter_duration,
        mode: 'lines',
        name: 'Inverter'

    };
    const trace5 = {
        x: pv_percentage,
        y: battery_charge_duration,
        mode: 'lines',
        name: 'Battery - Charging'

    };
    const trace6 = {
        x: pv_percentage,
        y: battery_discharge_duration,
        mode: 'lines',
        name: 'Battery - Discharging'

    };

    var data = [trace1, trace2, trace3, trace4, trace5, trace6];

    const layout = {
        plot_bgcolor: '#FAFAFA',
        paper_bgcolor: '#FAFAFA',
        xaxis: {
            title: 'Percentage of Operation in %',
            titlefont: {
                size: 16,
            },
            tickfont: {
                size: 14,
            },
        },
        yaxis: {
            title: 'Load in %',
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

// DEMAND COVERAGE PLOT
function plot_co2_emissions(co2_emissions) {


    const { non_renewable_electricity_production, hybrid_electricity_production} = co2_emissions;
    const time = Array.from({ length: non_renewable_electricity_production.length }, (_, i) => i);
    const non_renewable = non_renewable_electricity_production;
    const hybrid = hybrid_electricity_production;


    const xAxisTitle = time.length > 366 ? 'Time in hours' : 'Time in days';
    const co2Emissions = document.getElementById("co2Emissions");
    const trace1 = {
        x: time,
        y: non_renewable,
        mode: 'lines',
        name: 'Non-Renewable'
    };
    const trace2 = {
        x: time,
        y: hybrid,
        mode: 'none',
        fill: 'tonexty',
        name: 'Savings'
    };
    const trace3 = {
        x: time,
        y: hybrid,
        mode: 'lines',
        name: 'Hybrid'
    };
    var data = [trace1, trace2, trace3];
    const layout = {
        plot_bgcolor: '#FAFAFA',
        paper_bgcolor: '#FAFAFA',
        xaxis: {
            title: xAxisTitle,
            titlefont: {
                size: 16,
            },
            tickfont: {
                size: 14,
            },
        },
        yaxis: {
            title: 'CO<sub>2</sub> Emissions [tons]',
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

async function redirect(href) {
    window.location.href = href;
}

async function hide_grid_results() {
    // Hide the GRID section
    const gridSubtitle = document.getElementById('gridTitle'); // Find the subtitle element
    const gridRow = document.getElementById('gridResultsRow');
    if (gridSubtitle) {
        gridSubtitle.style.display = 'none'; // Hide the subtitle
    }
    if (gridRow && gridRow.classList.contains('row')) {
        gridRow.style.display = 'none'; // Hide the row associated with the GRID subtitle
    }
    // Now perform the row swap
    const row1 = document.getElementById('actionButtonsRow'); // Assuming this is the row with action buttons
    const row2 = document.getElementById('resultsChart'); // The row with results chart
    // Get the parent element of the rows
    const parentElement = row1.parentElement;
    // Ensure both rows exist before attempting to swap
    if (row1 && row2 && parentElement) {
        // Swap the rows using insertBefore
        parentElement.insertBefore(row2, row1);
    }
    hideElements("firstRow");
}




async function hide_es_results() {
    hideElements('resultsChart');
    hideElements('demandcoverageChart');
    hideElements('energyflowsChart');
    hideElements('capacityChart');
    hideElements('durationcurveChart');
    hideElements('sankeyChart');
        }

function hideElements(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.display = 'none';
    }
}

async function replaceSummaryChart() {
    // Get the existing chart element
    const summaryChart = document.getElementById('summaryResultsChart');

    // Define a list of ID pairs without and with a '2'
    const idPairs = [
        { original: 'nConsumers', newId: 'nConsumers2' },
        { original: 'nShsConsumers', newId: 'nShsConsumers2' },
        { original: 'nPoles', newId: 'nPoles2' },
        { original: 'lengthDistributionCable', newId: 'lengthDistributionCable2' },
        { original: 'averageLengthDistributionCable', newId: 'averageLengthDistributionCable2' },
        { original: 'lengthConnectionCable', newId: 'lengthConnectionCable2' },
        { original: 'averageLengthConnectionCable', newId: 'averageLengthConnectionCable2' },
        { original: 'GridUpfrontInvestmentCost', newId: 'GridUpfrontInvestmentCost2' },
        { original: 'time', newId: 'time2' }
    ];

    // Prepare an object to hold old values
    const values = {};

    // Retrieve current values and store them
    idPairs.forEach(pair => {
        const originalElement = document.getElementById(pair.original);
        if (originalElement) {
            values[pair.newId] = originalElement.innerHTML; // Store old value with new ID key
        } else {
            values[pair.newId] = ''; // Default empty if not found
        }
    });

    // Define the new content with modified IDs and set values from the old content
    const newContent = `
        <div class="chart" id="summaryResultsChartGridOnly">
            <div class="chart__header">
                <span class="title">Summary of Results</span>
            </div>
            <div class="chart__content">
                <span class="subtitle"></span>
                <div class="row">
                    <div class="item item--best">
                        <div class="item__name">Number of grid-connected Consumers</div>
                        <div id="nConsumers2" class="item__value">${values.nConsumers2}</div>
                    </div>
                    <div class="item item--best">
                        <div class="item__name">Number of SHS Consumers</div>
                        <div id="nShsConsumers2" class="item__value">${values.nShsConsumers2}</div>
                    </div>
                    <div class="item item--best">
                        <div class="item__name">Number of Poles</div>
                        <div id="nPoles2" class="item__value">${values.nPoles2}</div>
                    </div>
                    <div class="item item--best">
                        <div class="item__name">Distribution Cable Length</div>
                        <div id="lengthDistributionCable2" class="item__value">${values.lengthDistributionCable2}</div>
                    </div>
                </div>
                <span class="subtitle"></span>
                <div class="row">
                    <div class="item item--best">
                        <div class="item__name">Average Length Distribution</div>
                        <div id="averageLengthDistributionCable2" class="item__value">${values.averageLengthDistributionCable2}</div>
                    </div>
                    <div class="item item--best">
                        <div class="item__name">Connection Cable Length</div>
                        <div id="lengthConnectionCable2" class="item__value">${values.lengthConnectionCable2}</div>
                    </div>
                    <div class="item item--best">
                        <div class="item__name">Average Length Connection</div>
                        <div id="averageLengthConnectionCable2" class="item__value">${values.averageLengthConnectionCable2}</div>
                    </div>
                    <div class="item item--best">
                        <div class="item__name">Grid upfront Investment Cost</div>
                        <div id="GridUpfrontInvestmentCost2" class="item__value">${values.GridUpfrontInvestmentCost2}</div>
                    </div>
                </div>
                <span class="subtitle"></span>
                <div class="row">
                    <div class="item item--best">
                        <div class="item__name"></div>
                        <div class="item__value"></div>
                    </div>
                    <div class="item item--best">
                        <div class="item__name"></div>
                        <div class="item__value"></div>
                    </div>
                    <div class="item item--best">
                        <div class="item__name"></div>
                        <div class="item__value"></div>
                    </div>
                    <div class="item item--worst">
                        <div class="item__name">Calculation Time</div>
                        <div id="time2" class="item__value">${values.time2}</div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Replace the existing chart with the new content
    summaryChart.outerHTML = newContent;
}
