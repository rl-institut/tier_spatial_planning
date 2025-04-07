// Flag to track if a download is in progress
let isDownloadingCSV = false;

document.getElementById('downloadCSV').addEventListener('click', function (event) {
    event.preventDefault();

    // Check if a download is already in progress
    if (isDownloadingCSV) {
        // Optionally, inform the user
        alert('A download is already in progress. Please wait.');
        return; // Exit the function to prevent multiple downloads
    }

    // Set the flag to indicate a download is in progress
    isDownloadingCSV = true;

    const downloadButton = this;
    const originalButtonText = downloadButton.innerHTML;

    // Disable the button visually and functionally
    downloadButton.style.pointerEvents = 'none'; // Prevent further clicks
    downloadButton.style.opacity = '0.6'; // Make it look disabled
    downloadButton.innerHTML = 'Processing...';

    // Allow the UI to update before starting the download
    requestAnimationFrame(() => {
        (async () => {
            try {
                // Fetch the CSV file
                const response = await fetch('/download_data/' + project_id + '/csv');
                if (!response.ok) {
                    throw new Error(`Network response was not ok: ${response.statusText}`);
                }

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);

                // Create a temporary link to trigger the download
                const a = document.createElement('a');
                a.href = url;
                a.download = `offgridplanner_data.xlsx`;
                document.body.appendChild(a);
                a.click();

                // Clean up
                a.remove();
                window.URL.revokeObjectURL(url);

                // Change button text to 'Downloading...'
                downloadButton.innerHTML = 'Downloading...';

                // Re-enable the button after a delay (e.g., 2 minutes)
                setTimeout(() => {
                    downloadButton.innerHTML = originalButtonText;
                }, 5000); // Delay of 120,000 milliseconds (2 minutes)

                // Re-enable the button after a delay (e.g., 2 minutes)
                setTimeout(() => {
                    // Reset the flag and re-enable the button
                    isDownloadingPDF = false;
                    downloadButton.style.pointerEvents = '';
                    downloadButton.style.opacity = '';
                }, 120000); // Delay of 120,000 milliseconds (2 minutes)
            } catch (error) {
                console.error('Error downloading CSV:', error);
                alert('An error occurred while downloading the XLSX file. Please try again.');

                // Reset the flag and re-enable the button immediately
                isDownloadingCSV = false;
                downloadButton.style.pointerEvents = '';
                downloadButton.style.opacity = '';
                downloadButton.innerHTML = originalButtonText;
            }
        })();
    });
});


// Flag to track if a PDF download is in progress
let isDownloadingPDF = false;

document.getElementById('downloadPDF').addEventListener('click', function (event) {
    event.preventDefault();

    // Check if a download is already in progress
    if (isDownloadingPDF) {
        alert('A download is already in progress. Please wait.');
        return; // Exit the function to prevent multiple downloads
    }

    // Set the flag to indicate a download is in progress
    isDownloadingPDF = true;

    const downloadButton = this;
    const originalButtonText = downloadButton.innerHTML;

    // Disable the button visually and functionally
    downloadButton.style.pointerEvents = 'none'; // Prevent further clicks
    downloadButton.style.opacity = '0.6'; // Make it look disabled
    downloadButton.innerHTML = 'Processing...';

    // Use setTimeout to allow the UI to update before heavy computations
    setTimeout(() => {
        (async () => {
            try {
                const plotIds = [];
                if (steps[0]) plotIds.push('demandTs');
                if (steps[1]) plotIds.push('map');
                if (steps[2]) plotIds.push('sankeyDiagram', 'energyFlows', 'lcoeBreakdown', 'demandCoverage');


                // Generate images (ensure this function is asynchronous)
                const images = await generateImages(plotIds);

                // Filter out null values
                const validImages = images.filter(img => img !== null);
                if (validImages.length === 0) {
                    console.warn('No valid images to send.');
                    alert('No valid images were generated. Please try again.');
                    throw new Error('No valid images generated.');
                }

                // Send images to the backend
                await sendImagesToBackend(validImages);

                // Change button text to 'Downloading...'
                downloadButton.innerHTML = 'Downloading...';
                setTimeout(() => {
                    downloadButton.innerHTML = originalButtonText;
                }, 5000); // Delay of 120,000 milliseconds (2 minutes)

                // Re-enable the button after a delay (e.g., 2 minutes)
                setTimeout(() => {
                    // Reset the flag and re-enable the button
                    isDownloadingPDF = false;
                    downloadButton.style.pointerEvents = '';
                    downloadButton.style.opacity = '';
                }, 120000); // Delay of 120,000 milliseconds (2 minutes)

            } catch (error) {
                console.error('Error generating PDF:', error);
                alert('An error occurred while generating the PDF. Please try again.');

                // Reset the flag and re-enable the button immediately
                isDownloadingPDF = false;
                downloadButton.style.pointerEvents = '';
                downloadButton.style.opacity = '';
                downloadButton.innerHTML = originalButtonText;
            }
        })();
    }, 0); // Delay of 0 milliseconds
});



function generateImages(plotIds) {
    const imagePromises = plotIds.map(plotId => {
        const plotElement = document.getElementById(plotId);
        if (!plotElement) {
            console.warn(`Plot element with ID '${plotId}' was not found.`);
            return Promise.resolve(null);
        }

        if (plotId === "map") {
            // Existing code for generating map image
            return generateMapImage(map) // Ensure 'map' is defined
                .then(function(imageData) {
                    return { id: plotId, data: imageData };
                })
                .catch(function(error) {
                    console.error(`Error generating image for ${plotId}:`, error);
                    return null;
                });
        } else if (plotId === 'energyFlows' || plotId === 'demandCoverage' || plotId === 'sankeyDiagram') {
            // For these plots, we need to clone and adjust data, x-axis, and legend

            // Clone the plot data and layout
            const clonedData = JSON.parse(JSON.stringify(plotElement.data));
            const clonedLayout = JSON.parse(JSON.stringify(plotElement.layout));

            // Specific adjustments based on plotId
            if (plotId === 'sankeyDiagram') {
                // 1. Change margin top and bottom to 30
                clonedLayout.margin = clonedLayout.margin || {};
                clonedLayout.margin.t = 30;
                clonedLayout.margin.b = 30;

                // 2. Reduce height by 33% (set to 67% of original)
                if (clonedLayout.height) {
                    clonedLayout.height = clonedLayout.height * 0.5;
                } else {
                    // If height is not defined, set a default height reduced by 33%
                    clonedLayout.height = 600 * 0.5; // Example: original height = 600
                }
            } else if (plotId === 'energyFlows' || plotId === 'demandCoverage') {
                // Reduce height to 80% of original
                if (clonedLayout.height) {
                    clonedLayout.height = clonedLayout.height * 0.80;
                } else {
                    // If height is not defined, set a default height reduced to 80%
                    clonedLayout.height = 400; // Example: original height = 600
                }
            }

            // Determine the x-axis range
            let maxX = 0;
            clonedData.forEach(trace => {
                if (trace.x && trace.x.length > 0) {
                    const traceMaxX = Math.max(...trace.x);
                    if (traceMaxX > maxX) {
                        maxX = traceMaxX;
                    }
                }
            });

            // Desired x-axis end point
            const desiredEnd = 168; // Adjusted as per your requirement

            // Adjust x-axis range(s)
            for (let axisName in clonedLayout) {
                if (axisName.startsWith('xaxis')) {
                    clonedLayout[axisName] = clonedLayout[axisName] || {};
                    // Set the x-axis range based on data availability
                    if (maxX >= desiredEnd) {
                        clonedLayout[axisName].range = [0, desiredEnd];
                    } else {
                        clonedLayout[axisName].range = [0, maxX];
                    }
                    clonedLayout[axisName].autorange = false; // Disable autorange
                }
            }

            // **New Part: Replace data between x=0 and x=24 with data from x=24 to x=48**
            clonedData.forEach(trace => {
                if (trace.x && trace.y) {
                    const x = trace.x;
                    const y = trace.y;

                    // Check if we have enough data to perform the replacement
                    const hasEnoughData = x.some(value => value >= 48);

                    if (hasEnoughData) {
                        // Create new arrays for x and y
                        const newY = [...y]; // Clone y to avoid modifying original

                        // Map x values between 0 and 24 to x + 24
                        for (let i = 0; i < x.length; i++) {
                            if (x[i] >= 0 && x[i] <= 24) {
                                // Find the index where x equals x[i] + 24
                                const targetX = x[i] + 24;
                                const targetIndex = x.indexOf(targetX);
                                if (targetIndex !== -1) {
                                    // Replace y value at current index with y value from target index
                                    newY[i] = y[targetIndex];
                                }
                            }
                        }
                        // Assign the modified y-values back to the trace
                        trace.y = newY;
                    } else {
                        console.warn(`Not enough data to replace values for ${plotId}.`);
                    }
                }
            });

            // Create a hidden div
            const tempDiv = document.createElement('div');
            tempDiv.style.display = 'none';
            document.body.appendChild(tempDiv);

            // Render the cloned plot into the hidden div
            return Plotly.newPlot(tempDiv, clonedData, clonedLayout).then(function() {
                // Generate the image
                return Plotly.toImage(tempDiv, { format: 'svg' })
                    .then(function(imageData) {
                        return { id: plotId, data: imageData };
                    })
                    .catch(function(error) {
                        console.error(`Error generating image for ${plotId}:`, error);
                        return null;
                    })
                    .finally(function() {
                        // Clean up
                        Plotly.purge(tempDiv);
                        tempDiv.parentNode.removeChild(tempDiv);
                    });
            }).catch(function(error) {
                console.error(`Error rendering cloned plot for ${plotId}:`, error);
                // Clean up
                Plotly.purge(tempDiv);
                tempDiv.parentNode.removeChild(tempDiv);
                return null;
            });
        } else {
            // For other plots, proceed as usual
            return Plotly.toImage(plotElement, { format: 'svg' })
                .then(function(imageData) {
                    return { id: plotId, data: imageData };
                })
                .catch(function(error) {
                    console.error(`Error generating image for ${plotId}:`, error);
                    return null;
                });
        }
    });

    return Promise.all(imagePromises);
}






function generateMapImage(map) {
    return new Promise((resolve, reject) => {
        if (!map || typeof map.getContainer !== 'function') {
            console.error('Map-Objekt ist nicht definiert oder ungültig.');
            return reject(new Error('Ungültiges Map-Objekt.'));
        }

        const mapContainer = map.getContainer();

        // Select elements you want to hide (adjust selectors as needed)
        const zoomControl = document.querySelector('.leaflet-control-zoom');
        const layerControl = document.querySelector('.leaflet-control-layers');
        const customZoomButton = document.querySelector('.leaflet-control-custom');

        // Hide elements before capturing
        if (zoomControl) zoomControl.style.display = 'none';
        if (layerControl) layerControl.style.display = 'none';
        if (customZoomButton) customZoomButton.style.display = 'none';

        const fixedWidth = 1600;
        const fixedHeight = 800;

        html2canvas(mapContainer, {
            useCORS: true,
            allowTaint: true,
            logging: false,
            backgroundColor: null,
            scale: 1,
            windowWidth: fixedWidth,
            windowHeight: fixedHeight,
            scrollX: -window.scrollX,
            scrollY: -window.scrollY
        })
        .then(canvas => {
            const imgData = canvas.toDataURL('image/png');
            resolve(imgData);

            // Restore the visibility of hidden elements after capturing
            if (zoomControl) zoomControl.style.display = '';
            if (layerControl) layerControl.style.display = '';
            if (customZoomButton) customZoomButton.style.display = '';
        })
        .catch(err => {
            console.error('Fehler beim Generieren des Kartenbildes mit html2canvas:', err);
            reject(err);

            // Restore visibility if there's an error
            if (zoomControl) zoomControl.style.display = '';
            if (layerControl) layerControl.style.display = '';
            if (customZoomButton) customZoomButton.style.display = '';
        });
    });
}


function sendImagesToBackend(images) {
    const data = {
        images: images  // Array of { id: plotId, data: imageData }
    };
    fetch(`/download_pdf_report/` + project_id, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.blob(); // Get the response as a blob (PDF file)
    })
    .then(blob => {
        // Create a URL for the blob
        const url = window.URL.createObjectURL(blob);
        // Create a temporary link to trigger the download
        const a = document.createElement('a');
        a.href = url;
        a.download = `offgridplanner_results.pdf`;
        document.body.appendChild(a);
        a.click();
        // Clean up
        a.remove();
        window.URL.revokeObjectURL(url);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}


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
    let xValue = [
        'PV',
        'Inverter',
        'Rectifier',
        'Diesel Genset',
        'Peak Demand',
        'Max. Surplus',
        'Battery'
    ];

    // Reverse the arrays
    xValue = xValue.reverse();
    yValue = yValue.reverse();
    let colors = [
        'rgb(8,48,107)', 'rgb(8,48,107)', 'rgb(8,48,107)', 'rgb(8,48,107)',
        'rgb(8,48,107)', 'rgb(8,48,107)', 'rgb(133, 52, 124)'
    ];
    colors = colors.reverse();  // Reverse the color array

    var dataTraces = [
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
            y: ['Battery'],
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
            automargin: true, // Enable automatic margin adjustment
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
        autosize: true,  // Enable automatic sizing
        margin: {
            l: 80,  // Reduced left margin
            r: 50,
            b: 100,
            t: 100,
            pad: 4
        },
    };

    Plotly.newPlot(optimalSizes, dataTraces, layout);
};



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
            x: 0.5,           // Positions the legend horizontally at the center (50% of the plot width)
            y: 1.15,          // Positions the legend vertically above the plot area
            xanchor: 'center',// Anchors the legend horizontally at its center
            yanchor: 'bottom',// Anchors the legend vertically at the bottom
            orientation: 'h', // Sets the legend items to be displayed horizontally
            bgcolor: 'rgba(255, 255, 255, 1)', // Fully opaque white background
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
        legend: {
            x: 0.5,           // Positions the legend horizontally at the center (50% of the plot width)
            y: 1.15,          // Positions the legend vertically above the plot area
            xanchor: 'center',// Anchors the legend horizontally at its center
            yanchor: 'bottom',// Anchors the legend vertically at the bottom
            orientation: 'h', // Sets the legend items to be displayed horizontally
            bgcolor: 'rgba(255, 255, 255, 1)', // Fully opaque white background
            bordercolor: '#E2E2E2',
            borderwidth: 2,
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

function plot_demand_24h(data) {
    let demandTs = document.getElementById("demandTs");

    var layout = {
        font: { size: 14 },
        autosize: true,
        xaxis: {
            title: 'Hour of the day',
            hoverformat: '.1f',
            titlefont: { size: 16 },
            tickfont: { size: 14 },
        },
        yaxis: {
            title: 'Demand (kW)',
            hoverformat: '.1f',
            titlefont: { size: 16 },
            tickfont: { size: 14 },
        },
        legend: {
            x: 0.5,
            y: 1.15,
            xanchor: 'center',
            yanchor: 'bottom',
            orientation: 'h',
            bgcolor: 'rgba(255, 255, 255, 1)',
            bordercolor: '#E2E2E2',
            borderwidth: 2,
        }
    };

    // Extract data from the passed-in data object
    let {
        x,
        households,
        enterprises,
        public_services
    } = data;

    // Define traces with 'stackgroup'
    var traceHouseholds = {
        x: x,
        y: households,
        type: 'scatter',
        mode: 'lines',
        name: 'Demand of Households',
        line: { shape: 'spline', width: 2, color: 'rgba(31, 119, 180, 1)' },
        fill: 'tonexty',
        fillcolor: 'rgba(31, 119, 180, 0.5)',
        stackgroup: 'one' // Group for stacking
    };

    var traceEnterprises = {
        x: x,
        y: enterprises,
        type: 'scatter',
        mode: 'lines',
        name: 'Demand of Enterprises',
        line: { shape: 'spline', width: 2, color: 'rgba(255, 127, 14, 1)' },
        fill: 'tonexty',
        fillcolor: 'rgba(255, 127, 14, 0.5)',
        stackgroup: 'one' // Same group as above
    };

    var tracePublicServices = {
        x: x,
        y: public_services,
        type: 'scatter',
        mode: 'lines',
        name: 'Demand of Public Services',
        line: { shape: 'spline', width: 2, color: 'rgba(44, 160, 44, 1)' },
        fill: 'tonexty',
        fillcolor: 'rgba(44, 160, 44, 0.5)',
        stackgroup: 'one' // Same group as above
    };

    // Data array
    var dataTraces = [tracePublicServices, traceEnterprises, traceHouseholds];

    // Render plot with the traces
    Plotly.react(demandTs, dataTraces, layout);
}


