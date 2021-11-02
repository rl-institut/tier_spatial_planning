async function importConfig(include_settings) {
  let formData = new FormData();
  formData.append("file", config_import.files[0]);

  $.ajax({
    url: "import_config",
    type: "POST",
    data: formData,
    processData: false,
    contentType: false,
    statusCode: {
      200: function (result) {
        // refreshNodeFromDataBase();
        // refreshLinksFromDatBase();
        database_to_map(nodes_or_links = 'nodes');
        database_to_map(nodes_or_links = 'links');

        if (include_settings === true) {
          setSettings(
            (cost_pole = result.cost_pole),
            (cost_connection = result.cost_connection),
            (cost_interpole_cable = result.cost_interpole_cable),
            (cost_distribution_cable = result.cost_distribution_cable),
            (shs_identification_connection_price =
              result.shs_identification_connection_price),
            (number_of_relaxation_steps_nr =
              result.number_of_relaxation_steps_nr)
          );
        }
      },
    },
  });
}

function setSettings(
  cost_pole,
  cost_connection,
  cost_interpole_cable,
  cost_distribution_cable,
  shs_identification_cable_price,
  shs_identification_connection_price,
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

async function generateExportFile() {
  $.ajax({
    url: "generate_export_file/",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      cost_pole: cost_pole.value,
      cost_connection: cost_connection.value,
      cost_interpole_cable: cost_interpole_cable.value,
      cost_distribution_cable: cost_distribution_cable.value,
      shs_identification_cable_price: cost_distribution_cable.value,
      //      shs_identification_cable_price:
      //        cable_price_per_meter_for_shs_mst_identification.value,
      //shs_identification_connection_price:
      //additional_connection_price_for_shs_mst_identification.value,
      shs_identification_connection_price: 0,
      number_of_relaxation_steps_nr: number_of_relaxation_steps_nr.value,
    }),
    dataType: "json",
    statusCode: {
      200: function () {
        downloadExportFile();
      },
    },
  });
}

function downloadExportFile() {
  window.open("download_export_file", "_self");
}
