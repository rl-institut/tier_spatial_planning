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

$(function() {
    $("input[name='options2']").change(function() {
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

document.addEventListener('DOMContentLoaded', function() {
  for (let i = 0; i < radioButtons.length; i++) {
    radioButtons[i].addEventListener("change", function() {
      if (this.checked) {
        selectedValue = i;
      }
    });
  }
});



function demand_ts(project_id ) {
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

          var plotElement = document.getElementById("demand_plot");
          var trace = {
              x: x,
              y: y,
              mode: 'line',
              name: 'Random Data',
              line: {
                    color: 'blue',
                    width: 2.5
              },
          };

          var layout = {
            xaxis: {
              title: 'X',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              },
            },
            yaxis: {
              title: 'Y',
              titlefont: {
                size: 16,
              },
              tickfont: {
                size: 14,
              }
            },
          };

          var data = [trace];

          Plotly.newPlot(plotElement, data, layout);
      }
  };
}

