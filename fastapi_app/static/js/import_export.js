/*function handleFile(e) {
  var file = e.target.files[0];
  var reader = new FileReader();
  reader.onload = (function (f) {
    return function (e) {
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
      //return import_file_content;
    };
  })(file);
  reader.readAsArrayBuffer(file);

}
*/

//document.getElementById('import').addEventListener('change', (event) => {
//  selected_file = event.target.files[0];
//})

function import_data() {
  // click on the hidden input file button
  document.getElementById('import').click();

  // when the element's status gets changed, the following code will be executed
  var element = document.getElementById('import');
  element.addEventListener('change', (event) => {
    let selected_file = event.target.files[0];
    let file_reader = new FileReader();
    file_reader.readAsBinaryString(selected_file);
    file_reader.onload = (event) => {
      let import_data = event.target.result;
      let workbook = XLSX.read(import_data, { type: "binary" });

      // import settings to the web app
      let settings_row_object = XLSX.utils.sheet_to_row_object_array(workbook.Sheets['settings'], {
        blankrows: false,
        header: 1,
        raw: true,
        rawNumbers: true
      });
      settings_row_object.shift(); // remove the first array only containing the sheet name and "value"
      settings_dict = settings_row_object.reduce((dict, [key, value]) => Object.assign(dict, { [key]: value }), {}); // converting the array to dictionary
      import_settings_to_webapp(settings_dict);

      // copy nodes and links into the existing *.csv files
      let nodes_to_import = XLSX.utils.sheet_to_row_object_array(workbook.Sheets['nodes']);
      let links_to_import = XLSX.utils.sheet_to_row_object_array(workbook.Sheets['links']);
      nodes_to_import_json = JSON.stringify(nodes_to_import);

      $("#loading").show();
      $.ajax({
        url: "/import_data",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({
          nodes_to_import,
          links_to_import
        }),
        dataType: "json",
        statusCode: {
          200: function () {
            database_to_map(nodes_or_links = 'nodes');
            database_to_map(nodes_or_links = 'links');
            $("#loading").hide();
          },
        },
      });

      //console.log(settings_json_file);
    }
  });
}

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
//}

function import_settings_to_webapp(settings_dict) {
  document.getElementById("cost_pole").value = settings_dict.cost_pole;
  document.getElementById("cost_connection").value = settings_dict.cost_connection;
  document.getElementById("cost_interpole_cable").value = settings_dict.cost_interpole_cable;
  document.getElementById("cost_distribution_cable").value = settings_dict.cost_distribution_cable;
  document.getElementById("number_of_relaxation_steps_nr").value = settings_dict.number_of_relaxation_steps_nr;
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