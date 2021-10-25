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
        refreshNodeFromDataBase();
        refreshLinksFromDatBase();
        if (include_settings === true) {
          setSettings(
            (price_pole = result.price_pole),
            (price_consumer = result.price_consumer),
            (price_pole_cable = result.price_pole_cable),
            (price_distribution_cable = result.price_distribution_cable),
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
  price_pole,
  price_consumer,
  price_pole_cable,
  price_distribution_cable,
  shs_identification_cable_price,
  shs_identification_connection_price,
  number_of_relaxation_steps_nr
) {
  document.getElementById("price_pole").value = price_pole;
  document.getElementById("price_consumer").value = price_consumer;
  document.getElementById("price_pole_cable").value = price_pole_cable;
  document.getElementById("price_distribution_cable").value =
    price_distribution_cable;
  document.getElementById("number_of_relaxation_steps_nr").value =
    number_of_relaxation_steps_nr;
}

async function generateExportFile() {
  $.ajax({
    url: "generate_export_file/",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      price_pole: price_pole.value,
      price_consumer: price_consumer.value,
      price_pole_cable: price_pole_cable.value,
      price_distribution_cable: price_distribution_cable.value,
      shs_identification_cable_price: price_distribution_cable.value,
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
