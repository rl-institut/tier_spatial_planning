/************************************************************/
/*                          IMPORT                          */
/************************************************************/

// import nodes, links, and settings from the selected excel file
function import_data() {
  // click on the hidden input file button (to change the status of the element)
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
      settings_row_object.shift(); // remove the first array only containing the sheet name ("settings") and "value"
      settings_dict = settings_row_object.reduce((dict, [key, value]) => Object.assign(dict, { [key]: value }), {}); // convert the array to dictionary
      import_settings_to_webapp(settings_dict);

      // copy nodes and links into the existing *.csv files
      let nodes_to_import = XLSX.utils.sheet_to_row_object_array(workbook.Sheets['nodes']);
      let links_to_import = XLSX.utils.sheet_to_row_object_array(workbook.Sheets['links']);
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
    }
  });
}


// put the settings into the web app
function import_settings_to_webapp(settings_dict) {
  document.getElementById("cost_pole").value = settings_dict.cost_pole;
  document.getElementById("cost_connection").value = settings_dict.cost_connection;
  document.getElementById("cost_interpole_cable").value = settings_dict.cost_interpole_cable;
  document.getElementById("cost_distribution_cable").value = settings_dict.cost_distribution_cable;
  document.getElementById("number_of_relaxation_steps_nr").value = settings_dict.number_of_relaxation_steps_nr;
}


/************************************************************/
/*                          EXPORT                          */
/************************************************************/

// export nodes, links, and settings to the selected excel file
async function export_data() {
  $.ajax({
    url: "export_data/",
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


function download(text, name, type) {
  var default_file_name = 'import_export'
  var file_name = prompt('Enter the file name:', default_file_name);

  var button_export = document.getElementById("a");
  var file = new Blob([text], { type: type });
  a.href = URL.createObjectURL(file);
  a.download = name;
}


function download_file() {
  var FileSaver = require('file-saver');
  var blob = new Blob(["Hello, world!"], { type: "text/plain;charset=utf-8" });
  FileSaver.saveAs(blob, "hello world.txt");
}
