/**
 * This script handles UI interactions and data visualization in a web application.
 * - Toggles visibility of an accordion section based on a switch.
 * - Adjusts input fields in response to radio button selections.
 * - Listens for radio button changes and stores the selected value.
 * - Fetches and plots time series data for demand profiles using Plotly.
 */


document.getElementById('toggleswitch').addEventListener('change', function (event) {
    const accordionItem2 = new bootstrap.Collapse(document.getElementById('collapseTwo'), {
        toggle: false
    });
    if (event.target.checked) {
        accordionItem2.show();
    } else {
        accordionItem2.hide();
    }
});

document.getElementById('toggleswitch2').addEventListener('change', function (event) {
    const accordionItem1 = new bootstrap.Collapse(document.getElementById('collapseOne'), {toggle: false});
    const accordionItem2 = document.getElementById('collapseTwo').closest('.accordion-item');
    const accordionItem3 = new bootstrap.Collapse(document.getElementById('collapseThree'), {toggle: false});
    const accordionItem3_all = document.getElementById('collapseThree').closest('.accordion-item');
    if (event.target.checked) {
        accordionItem1.hide();
        accordionItem2.style.display = 'none';
        accordionItem3.show();
        accordionItem3_all.style.display = 'block';
    } else {
        accordionItem1.show();
        accordionItem2.style.display = 'block';
        accordionItem3.hide();
        accordionItem3_all.style.display = 'none';
    }
});

$(function () {
    $("input[name='options2']").change(function () {
        if ($("#option7").is(':checked')) {
            $("#average_daily_energy").prop('disabled', false);
            $("#maximum_peak_load").prop('disabled', true).val('');
        } else {
            $("#average_daily_energy").prop('disabled', true).val('');
            $("#maximum_peak_load").prop('disabled', false);
        }
    });
});

const radioButtons = document.getElementsByName("options");
let selectedValue = -1; // Default value if no selection is made

document.addEventListener('DOMContentLoaded', function () {
    for (let i = 0; i < radioButtons.length; i++) {
        radioButtons[i].addEventListener("change", function () {
            if (this.checked) {
                selectedValue = i;
            }
        });
    }
});


function demand_ts(project_id) {
    const url = 'get_demand_plot_data/' + project_id;
    let plotElement = document.getElementById("demand_plot");

    // Get references to the radio buttons
    var radioTotalDemand = document.getElementById('optionTotalDemand');
    var radioSingleHousehold = document.getElementById('optionSingleHousehold');

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
            orientation: 'h',
            x: 0,
            y: -0.3,
            xanchor: 'left',
            yanchor: 'top',
            traceorder: 'normal' // Ensure legendrank is honored
        }
    };

    // Initialize the plot with empty data
    Plotly.newPlot(plotElement, [], layout);

    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Extract data
            const {
                'x': x,
                'Very High Consumption': Very_High,
                'High Consumption': High,
                'Middle Consumption': Middle,
                'Low Consumption': Low,
                'Very Low Consumption': Very_Low,
                'National': National,
                'households': households,
                'enterprises': enterprises,
                'public_services': public_services
            } = data;

            // Compute Total_Demand
            var Total_Demand = households.map((value, index) => {
                return value + enterprises[index] + public_services[index];
            });

            // Define traces
            var trace10 = {
                x: x,
                y: Total_Demand,
                mode: 'lines',
                name: 'Total Demand',
                line: { color: 'black', width: 3, shape: 'spline' },
                visible: true, // Initially visible
                legendrank: 0
            };

            var trace7 = {
                x: x,
                y: households,
                type: 'scatter',
                mode: 'lines',
                name: 'Demand of Households',
                stackgroup: 'one',
                fill: 'tonexty',
                hoverinfo: 'x+y',
                line: { shape: 'spline', width: 0.5, color: 'rgba(31, 119, 180, 1)' },
                fillcolor: 'rgba(31, 119, 180, 0.6)',
                legendrank: 3
            };

            var trace8 = {
                x: x,
                y: enterprises,
                type: 'scatter',
                mode: 'lines',
                name: 'Demand of Enterprises',
                stackgroup: 'one',
                fill: 'tonexty',
                hoverinfo: 'x+y',
                line: { shape: 'spline', width: 0.5, color: 'rgba(255, 127, 14, 1)' },
                fillcolor: 'rgba(255, 127, 14, 0.6)',
                legendrank: 2
            };

            var trace9 = {
                x: x,
                y: public_services,
                type: 'scatter',
                mode: 'lines',
                name: 'Demand of Public Services',
                stackgroup: 'one',
                fill: 'tonexty',
                hoverinfo: 'x+y',
                line: { shape: 'spline', width: 0.5, color: 'rgba(44, 160, 44, 1)' },
                fillcolor: 'rgba(44, 160, 44, 0.6)',
                legendrank: 1
            };

            var trace6 = {
                x: x,
                y: National,
                mode: 'lines',
                name: 'Single Household Profile',
                line: { color: 'black', width: 2, shape: 'spline' },
                visible: false, // Initially hidden
                legendrank: 4
            };

            var trace5 = {
                x: x,
                y: Very_High,
                mode: 'lines',
                name: 'Very High Consumption',
                line: { color: 'blue', width: 1, shape: 'spline' },
                visible: 'legendonly',
                legendrank: 5
            };

            var trace4 = {
                x: x,
                y: High,
                mode: 'lines',
                name: 'High Consumption',
                line: { color: 'green', width: 1, shape: 'spline' },
                visible: 'legendonly',
                legendrank: 6
            };

            var trace3 = {
                x: x,
                y: Middle,
                mode: 'lines',
                name: 'Middle Consumption',
                line: { color: 'black', width: 1, shape: 'spline' },
                visible: 'legendonly',
                legendrank: 7
            };

            var trace2 = {
                x: x,
                y: Low,
                mode: 'lines',
                name: 'Low Consumption',
                line: { color: 'orange', width: 1, shape: 'spline' },
                visible: 'legendonly',
                legendrank: 8
            };

            var trace1 = {
                x: x,
                y: Very_Low,
                mode: 'lines',
                name: 'Very Low Consumption',
                line: { color: 'red', width: 1, shape: 'spline' },
                visible: 'legendonly',
                legendrank: 9
            };

            // Data array (order is important for stacking and layering)
            var dataTraces = [trace10, trace9, trace8, trace7, trace6, trace5, trace4, trace3, trace2, trace1];

            // Render plot with all traces
            Plotly.react(plotElement, dataTraces, layout);

            // Function to update plot based on selection
            function updatePlot() {
                if (radioTotalDemand.checked) {
                    // Activate traces 1 to 6 (indices 0 to 5)
                    Plotly.restyle(plotElement, { 'visible': true }, [0, 1, 2, 3]);
                    // Deactivate traces 7 to 10 (indices 6 to 9)
                    Plotly.restyle(plotElement, { 'visible': 'legendonly' }, [4, 5, 6, 7, 8, 9]);
                } else if (radioSingleHousehold.checked) {
                    // Activate traces 7 to 10 (indices 6 to 9)
                    Plotly.restyle(plotElement, { 'visible': true }, [4, 5, 6, 7, 8, 9]);
                    // Deactivate traces 1 to 6 (indices 0 to 5)
                    Plotly.restyle(plotElement, { 'visible': 'legendonly' }, [0, 1, 2, 3]);
                }
            }

            // Add event listeners to radio buttons
            radioTotalDemand.addEventListener('change', updatePlot);
            radioSingleHousehold.addEventListener('change', updatePlot);

            // Initial plot update based on default selection
            updatePlot();

        })
        .catch(error => {
            console.error('There was a problem with the fetch operation:', error);
        });
}






// Trigger the file input dialog when the "Import Consumers" button is clicked
document.getElementById('importButton').addEventListener('click', function() {
    document.getElementById('fileInput').click();
});

// Handle the file selection and upload the file to the server
document.getElementById('fileInput').addEventListener('change', async function(event) {
    const file = event.target.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);
        await file_demand_to_db(formData);
        document.getElementById('fileInput').value = '';
    }
});

document.getElementById('downloadDemand').addEventListener('click', function () {
    save_demand_estimation('javascript:void(0);')
    window.location.href = '/export_demand/' + project_id + '/' + document.getElementById('fileTypeDropdown').value+ '/';
});

function loadDashboard() {
    const dashboardSection = document.querySelector('.dashboard');

    // Check if the 'loading' class is not already present
    if (!dashboardSection.classList.contains('loading')) {
        dashboardSection.classList.add('loading');
    }
}
