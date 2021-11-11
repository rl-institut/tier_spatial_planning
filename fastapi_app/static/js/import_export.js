

function handleFile(e) {
  var file = e.target.files[0];
  var reader = new FileReader();
  reader.onload = function (e) {
    var data = new Uint8Array(e.target.result);
    var workbook = XLSX.read(data, { type: 'array' });
    var sheet_name_list = workbook.SheetNames;
    // sheet 1: nodes
    var nodes = XLSX.utils.sheet_to_json(workbook.Sheets[sheet_name_list[0]])
    // sheet 2: links
    var links = XLSX.utils.sheet_to_json(workbook.Sheets[sheet_name_list[1]])
    // sheet 1: settings
    var settings = XLSX.utils.sheet_to_json(workbook.Sheets[sheet_name_list[2]])

    var import_file_content = { "nodes": nodes, "links": links, "settings": settings };
  };
  reader.readAsArrayBuffer(file);
  return import_file_content;

}




function import_data() {
  element = document.getElementById('import');
  element.click();
  element.addEventListener('change', function (event) {
    var results = handleFile(event);
    result2 = results;
  }, false)
  //console.log('hi')
  /*
// read the excel file
var import_excel_file = document.getElementById('import');
import_excel_file.addEventListener('change', function () {
  readXlsxFile(import_excel_file.files[0]).then(function (data) {
    console.log(data)
  })
})
*/

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