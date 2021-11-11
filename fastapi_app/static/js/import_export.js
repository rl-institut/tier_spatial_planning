async function import_data(include_settings) {
  // trigger the click on the import button
  $(document.getElementById('import')).click();

  // read the excel file
  var import_excel_file = document.getElementById('import');
  import_excel_file.addEventListener('change', function () {
    readXlsxFile(import_excel_file.files[0]).then(function (data) {
      console.log(data)
    })
  })


  //let formData = new FormData();
  //formData.append("file", config_import.files[0]);

  /*
  $.ajax({
    url: "import_data",
    type: "POST",
    data: formData,
    processData: false,
    contentType: false,
    statusCode: {
      200: function (result) {
        database_to_map(nodes_or_links = 'nodes');
        database_to_map(nodes_or_links = 'links');

        if (include_settings === true) {
          import_settings_to_webapp(
            (cost_pole = result.cost_pole),
            (cost_connection = result.cost_connection),
            (cost_interpole_cable = result.cost_interpole_cable),
            (cost_distribution_cable = result.cost_distribution_cable),
            (shs_identification_cable_cost =
              result.shs_identification_cable_cost),
            (shs_identification_connection_cost = result.shs_identification_connection_cost),
            (number_of_relaxation_steps_nr =
              result.number_of_relaxation_steps_nr)
          );
        }
      },
    },
  });
  */
}

function import_settings_to_webapp(
  cost_pole,
  cost_connection,
  cost_interpole_cable,
  cost_distribution_cable,
  shs_identification_cable_cost,
  shs_identification_connection_cost,
  number_of_relaxation_steps_nr
) {
  document.getElementById("cost_pole").value = cost_pole;
  document.getElementById("cost_connection").value = cost_connection;
  document.getElementById("cost_interpole_cable").value = cost_interpole_cable;
  document.getElementById("cost_distribution_cable").value =
    cost_distribution_cable;
  document.getElementById("number_of_relaxation_steps_nr").value =
    number_of_relaxation_steps_nr;
}

async function generate_export_file() {
  $.ajax({
    url: "generate_export_file/",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      cost_pole: cost_pole.value,
      cost_connection: cost_connection.value,
      cost_interpole_cable: cost_interpole_cable.value,
      cost_distribution_cable: cost_distribution_cable.value,
      shs_identification_cable_cost: cost_distribution_cable.value,
      shs_identification_connection_cost: 0,
      number_of_relaxation_steps_nr: number_of_relaxation_steps_nr.value,
    }),
    dataType: "json",
  });
}