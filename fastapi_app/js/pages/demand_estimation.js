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
    var xhr = new XMLHttpRequest();
    url = 'get_demand_time_series/' + project_id;
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send()

    xhr.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            var data = this.response;
            var x = data['x'];
            var y = data['y'];
            var Very_High = data['Very High Consumption'];
            var High = data['High Consumption'];
            var Middle = data['Middle Consumption'];
            var Low = data['Low Consumption'];
            var Very_Low = data['Very Low Consumption'];
            var National = data['National'];
            var South_South = data['South South'];
            var North_West = data['North West'];
            var North_Central = data['North Central'];

            var plotElement = document.getElementById("demand_plot");

            var trace1 = {
                x: x,
                y: Very_Low,
                mode: 'line',
                name: 'Very Low Consumption (Category)',
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
                name: 'Low Consumption (Category)',
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
                name: 'Middle Consumption (Category)',
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
                name: 'High Consumption (Category)',
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
                name: 'Very High Consumption (Category)',
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
                name: 'National (Combination)',
                line: {
                    color: 'black',
                    width: 3,
                    shape: 'spline'
                },
            };
            var trace7 = {
                x: x,
                y: South_South,
                mode: 'line',
                name: 'South-South (Combination)',
                line: {
                    color: 'purple',
                    width: 3,
                    shape: 'spline'
                },
            };
            var trace8 = {
                x: x,
                y: North_West,
                mode: 'line',
                name: 'North-West (Combination)',
                line: {
                    color: 'light-blue',
                    width: 3,
                    shape: 'spline'
                },
            };
            var trace9 = {
                x: x,
                y: North_Central,
                mode: 'line',
                name: 'North-Central (Combination)',
                line: {
                    color: 'dark-red',
                    width: 3,
                    shape: 'spline'
                },
            };

            var layout = {
                title: "<b>Typical Modelled Household Daily Electrical Demand Profiles</b><br>'Average days' estimating <i>average contributions of each household</i> (to be scaled by community size)<br>365 days are modelled and included in profiles for simulation with full variability",
                font: {size: 14},
                autosize: false,
                width: 1100,
                height: 500,

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
            };

            var data = [trace9, trace8, trace7, trace6, trace5, trace4, trace3, trace2, trace1];

            Plotly.newPlot(plotElement, data, layout);
        }
    };
}

