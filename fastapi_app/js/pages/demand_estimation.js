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

function debounce(func, wait) {
    let timeout;
    return function(...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}



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
        if (steps) {
            steps[0] = false;
        }
    } else {
        accordionItem1.show();
        accordionItem2.style.display = 'block';
        accordionItem3.hide();
        accordionItem3_all.style.display = 'none';
        if (steps) {
            steps[0] = true;
        }
    }
});

// Adjust input fields based on radio button selection
document.addEventListener('DOMContentLoaded', function () {
    const option7Radio = document.getElementById('option7');
    const option8Radio = document.getElementById('option8');
    const averageDailyEnergyInput = document.getElementById('average_daily_energy');
    const maximumPeakLoadInput = document.getElementById('maximum_peak_load');

    function handleOptions2Change() {
        if (option7Radio.checked) {
            averageDailyEnergyInput.disabled = false;
            maximumPeakLoadInput.disabled = true;
            maximumPeakLoadInput.value = '';
        } else {
            averageDailyEnergyInput.disabled = true;
            averageDailyEnergyInput.value = '';
            maximumPeakLoadInput.disabled = false;
        }
    }

    option7Radio.addEventListener('change', handleOptions2Change);
    option8Radio.addEventListener('change', handleOptions2Change);

    // Initial setup
    handleOptions2Change();
});

// Store selected value from radio buttons with name 'options'
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
            let {
                'x': x,
                'Very High Consumption': Very_High,
                'High Consumption': High,
                'Middle Consumption': Middle,
                'Low Consumption': Low,
                'Very Low Consumption': Very_Low,
                'National': National,
                'households': households,
                'enterprises': enterprises,
                'public_services': public_services,
                'calibration_target_value': calibration_target_value,
                'calibration_option': calibration_option,
                'calibration_factor': calibration_factor,
                'num_households': num_households
            } = data;

            const enterprises_raw = enterprises.map(value => value / calibration_factor);
            const public_services_raw = public_services.map(value => value / calibration_factor);

            // Get references to the elements
            const toggleSwitch = document.getElementById('toggleswitch');
            const averageDailyEnergyInput = document.getElementById('average_daily_energy');
            const maximumPeakLoadInput = document.getElementById('maximum_peak_load');
            const option7Radio = document.getElementById('option7');
            const option8Radio = document.getElementById('option8');

            // Function to calculate Total_Demand
            function calculateTotalDemand(households, enterprises, public_services) {
                return households.map((value, index) => {
                    return value + enterprises[index] + public_services[index];
                });
            }

            // Usage inside your main function or logic
            var Total_Demand = calculateTotalDemand(households, enterprises, public_services);

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

            // Store trace1 to trace5 Y-values
            const trace1Y = dataTraces[9].y; // trace1: index 9
            const trace2Y = dataTraces[8].y; // trace2: index 8
            const trace3Y = dataTraces[7].y; // trace3: index 7
            const trace4Y = dataTraces[6].y; // trace4: index 6
            const trace5Y = dataTraces[5].y; // trace5: index 5

            // Get references to custom_share input fields
            const customShare1 = document.getElementById('custom_share_1');
            const customShare2 = document.getElementById('custom_share_2');
            const customShare3 = document.getElementById('custom_share_3');
            const customShare4 = document.getElementById('custom_share_4');
            const customShare5 = document.getElementById('custom_share_5');

            // Initialize an object to store previous values of custom share inputs
            const previousValues = {
                'custom_share_1': parseFloat(customShare1.value) || 0,
                'custom_share_2': parseFloat(customShare2.value) || 0,
                'custom_share_3': parseFloat(customShare3.value) || 0,
                'custom_share_4': parseFloat(customShare4.value) || 0,
                'custom_share_5': parseFloat(customShare5.value) || 0
            };

            // Function to update plot based on selection
            function showOnlySelection() {
                if (radioTotalDemand.checked) {
                    // Activate traces 1 to 4 (indices 0 to 3)
                    Plotly.restyle(plotElement, { 'visible': true }, [0, 1, 2, 3]);
                    // Deactivate traces 5 to 10 (indices 4 to 9)
                    Plotly.restyle(plotElement, { 'visible': 'legendonly' }, [4, 5, 6, 7, 8, 9]);
                } else if (radioSingleHousehold.checked) {
                    // Activate traces 5 to 10 (indices 4 to 9)
                    Plotly.restyle(plotElement, { 'visible': true }, [4, 5, 6, 7, 8, 9]);
                    // Deactivate traces 1 to 4 (indices 0 to 3)
                    Plotly.restyle(plotElement, { 'visible': 'legendonly' }, [0, 1, 2, 3]);
                }
            }

            function calibrate_demand(reverse = false) {

                var households_raw = National.map(value => value * num_households);

                let calibration_factor;
                const total_demand_raw = calculateTotalDemand(households_raw, enterprises_raw, public_services_raw);
                if (calibration_option === 'kW') {
                    calibration_factor = calibration_target_value / Math.max(...total_demand_raw);
                } else if (calibration_option === 'kWh') {
                    calibration_factor = calibration_target_value / total_demand_raw.reduce((a, b) => a + b, 0);
                } else {
                    calibration_factor = 1
                }
                households = households_raw.map(value => value * calibration_factor);
                enterprises = enterprises_raw.map(value => value * calibration_factor);
                public_services = public_services_raw.map(value => value * calibration_factor);
            }

            function updateTrace7to10() {
                Total_Demand = calculateTotalDemand(households, enterprises, public_services);
                // Restyle all traces in one command
                Plotly.restyle(plotElement, {
                    'y': [Total_Demand, public_services, enterprises, households]
                }, [0, 1, 2, 3]);
            }

            function updateNationalArray() {
                // Retrieve and parse input values, converting percentages to decimals
                const share1 = parseFloat(customShare1.value) / 100 || 0;
                const share2 = parseFloat(customShare2.value) / 100 || 0;
                const share3 = parseFloat(customShare3.value) / 100 || 0;
                const share4 = parseFloat(customShare4.value) / 100 || 0;
                const share5 = parseFloat(customShare5.value) / 100 || 0;

                National.forEach((val, idx) => {
                    National[idx] = (share1 * trace1Y[idx]) +
                                    (share2 * trace2Y[idx]) +
                                    (share3 * trace3Y[idx]) +
                                    (share4 * trace4Y[idx]) +
                                    (share5 * trace5Y[idx]);
                0});
            }

            // Attach event listener to reset button
            document.getElementById('resetDefault').addEventListener('click', resetToDefault);

            function resetToDefault() {
                // Set default values for custom shares
                customShare1.value = 66.3;
                customShare2.value = 21.5;
                customShare3.value = 7.6;
                customShare4.value = 3.1;
                customShare5.value = 1.5;

                // Update previousValues to match defaults
                previousValues['custom_share_1'] = 66.3;
                previousValues['custom_share_2'] = 21.5;
                previousValues['custom_share_3'] = 7.6;
                previousValues['custom_share_4'] = 3.1;
                previousValues['custom_share_5'] = 1.5;

                // Recalculate 'National' using default shares
                updateNationalArray();

                // Update trace6 (Single Household Profile)
                updateTrace6();

                // Reset calibration settings
                calibration_target_value = 1;
                calibration_option = null;

                // Recalculate 'households' based on new 'National' and 'num_households'
                households = National.map(value => value * num_households);

                // Reset 'enterprises' and 'public_services' to their raw values
                enterprises = enterprises_raw.slice(); // Make a copy to avoid modifying the original array
                public_services = public_services_raw.slice(); // Make a copy

                // Recalculate 'Total_Demand'
                Total_Demand = calculateTotalDemand(households, enterprises, public_services);

                // Update the plot for traces 0 to 3 (Total Demand, public services, enterprises, households)
                Plotly.restyle(plotElement, {
                    'y': [Total_Demand, public_services, enterprises, households]
                }, [0, 1, 2, 3]);
            }

            // Updates trace6 (Single Household Profile) based on custom share inputs
            function updateTrace6() {
                // Update trace6's Y-values in the Plotly plot
                Plotly.restyle(plotElement, { 'y': [National] }, [4]); // trace6 is at index 4
            }

            // Function to check if the change is significant (≥ 0.5)
            function isSignificantChange(newValue, oldValue, threshold = 0.2) {
                return Math.abs(newValue - oldValue) >= threshold;
            }

            // Function to handle input changes with threshold
            function handleInputChange(inputId) {
                return function () {
                    const input = document.getElementById(inputId);
                    const newValue = parseFloat(input.value) || 0;
                    const oldValue = previousValues[inputId];
                    if (isSignificantChange(newValue, oldValue)) {
                        previousValues[inputId] = newValue;
                        updateNationalArray()
                        updateTrace6();
                        calibrate_demand();
                        updateTrace7to10();
                    }
                };
            }

            // Add event listeners with threshold logic
            customShare1.addEventListener('input', handleInputChange('custom_share_1'), 250, false);
            customShare2.addEventListener('input', handleInputChange('custom_share_2'), 250, false);
            customShare3.addEventListener('input', handleInputChange('custom_share_3'), 250, false);
            customShare4.addEventListener('input', handleInputChange('custom_share_4'), 250, false);
            customShare5.addEventListener('input', handleInputChange('custom_share_5'), 250, false);

            // Add event listeners to radio buttons
            radioTotalDemand.addEventListener('change', showOnlySelection);
            radioSingleHousehold.addEventListener('change', showOnlySelection);

            // Function to handle calibration input changes
            function handleCalibrationInputChange() {
                // Only proceed if the toggle switch is activated
                if (toggleSwitch.checked) {
                    if (option7Radio.checked) {
                        // Option 7: "Set Average Total Annual Energy (kWh/year)"
                        const value = parseFloat(averageDailyEnergyInput.value);
                        if (!isNaN(value) && value >= 0) {
                            calibrate_demand(true);
                            calibration_target_value = value;
                            calibration_option = 'kWh';
                            calibrate_demand(false);
                            updateTrace7to10();
                        }
                    } else if (option8Radio.checked) {
                        // Option 8: "Set Maximum Peak Demand (kW)"
                        const value = parseFloat(maximumPeakLoadInput.value);
                        if (!isNaN(value) && value >= 0) {
                            calibrate_demand(true);
                            calibration_target_value = value;
                            calibration_option = 'kW';
                            calibrate_demand(false);
                            updateTrace7to10();
                        }
                    }
                } else {
                    // Toggle is deactivated
                    calibrate_demand(true);
                    calibration_target_value = 1;
                    calibration_option = null;
                    updateTrace7to10();
                    households = National.map(value => value * num_households);
                    calibrate_demand(false);
                }
            }

            // Add event listener to the toggle switch
            toggleSwitch.addEventListener('change', function(event) {
                if (!event.target.checked) {
                    // Toggle is deactivated
                    calibration_target_value = 1;
                    calibration_option = null;
                    updateTrace7to10();
                }
            });

            // Add event listeners to the radio buttons for calibration options
            function handleRadioButtonChange() {
                if (option7Radio.checked) {
                    averageDailyEnergyInput.disabled = false;
                    maximumPeakLoadInput.disabled = true;
                    maximumPeakLoadInput.value = '';
                    handleCalibrationInputChange();
                } else if (option8Radio.checked) {
                    averageDailyEnergyInput.disabled = true;
                    averageDailyEnergyInput.value = '';
                    maximumPeakLoadInput.disabled = false;
                    handleCalibrationInputChange();
                }
            }

            option7Radio.addEventListener('change', handleRadioButtonChange, 1, false);
            option8Radio.addEventListener('change', handleRadioButtonChange, 1, false);


            // Add event listeners to the calibration input fields with debounce
            averageDailyEnergyInput.addEventListener('input', debounce(handleCalibrationInputChange, 1000, false));
            maximumPeakLoadInput.addEventListener('input', debounce(handleCalibrationInputChange, 1000, false));



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
