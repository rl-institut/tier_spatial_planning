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
    const url = 'get_demand_time_series/' + project_id;
    let plotElement = document.getElementById("demand_plot");

    var layout = {
        title: "<b>Typical Modelled Household Daily Electrical Demand Profiles</b><br>'Average days' estimating <i>average contributions of each household</i> (to be scaled by community size)<br>365 days are modelled and included in profiles for simulation with full variability",
        font: {size: 14},
        autosize: true,
        xaxis: {
            title: 'Hour of the day',
            hoverformat: '.1f',
            titlefont: {
                size: 16,
            },
            tickfont: {
                size: 14,
            },
        },
        yaxis: {
            title: 'Demand (W)',
            hoverformat: '.1f',
            titlefont: {
                size: 16,
            },
            tickfont: {
                size: 14,
            }
        },
        legend: {
            orientation: 'h', // Set the legend to horizontal
            x: 0,
            y: -0.3, // Position the legend below the x-axis
            xanchor: 'left',
            yanchor: 'top',
        }
    };

    Plotly.newPlot(plotElement, [], layout);

    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Extracting data
            const {
                x,
                y,
                'Very High Consumption': Very_High,
                'High Consumption': High,
                'Middle Consumption': Middle,
                'Low Consumption': Low,
                'Very Low Consumption': Very_Low,
                National,
                'South South': South_South,
                'North West': North_West,
                'North Central': North_Central
            } = data;

            var trace1 = {
                x: x,
                y: Very_Low,
                mode: 'line',
                name: 'Very Low Consumption',
                line: {
                    color: 'red',
                    width: 1,
                    shape: 'spline'
                },
            };

            var trace2 = {
                x: x,
                y: Low,
                mode: 'line',
                name: 'Low Consumption',
                line: {
                    color: 'orange',
                    width: 1,
                    shape: 'spline'
                },
            };

            var trace3 = {
                x: x,
                y: Middle,
                mode: 'line',
                name: 'Middle Consumption',
                line: {
                    color: 'black',
                    width: 1,
                    shape: 'spline'
                },
            };

            var trace4 = {
                x: x,
                y: High,
                mode: 'line',
                name: 'High Consumption',
                line: {
                    color: 'green',
                    width: 1,
                    shape: 'spline'
                },
            };

            var trace5 = {
                x: x,
                y: Very_High,
                mode: 'line',
                name: 'Very High Consumption',
                line: {
                    color: 'blue',
                    width: 1,
                    shape: 'spline'
                },
            };

            var trace6 = {
                x: x,
                y: National,
                mode: 'line',
                name: 'Demand Profile',
                line: {
                    color: 'black',
                    width: 3,
                    shape: 'spline'
                },
            };

            var data = [trace6, trace5, trace4, trace3, trace2, trace1];

            Plotly.react(plotElement, data, layout);

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
