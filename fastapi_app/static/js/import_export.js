/************************************************************/
/*                          IMPORT                          */
/************************************************************/

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
    }
  });
}


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
// convert the binary data into octet, which is the correct content type for excel file
function binary_to_octet(string) {
  var buffer = new ArrayBuffer(string.length);
  var array = new Uint8Array(buffer);
  for (var i = 0; i < string.length; i++) array[i] = string.charCodeAt(i) & 0xFF;
  return array;
}


async function export_data() {
  // create the excel workbook and add some properties
  var workbook = XLSX.utils.book_new();
  workbook.Props = {
    Title: "Import and export of data to/from the web app.",
    Subject: "off-grid network and energy supply system",
    Author: "Saeed Sayadi",
    CreatedDate: new Date(2021, 11, 22)
  };

  // create sheets
  workbook.SheetNames.push("Nodes");
  var worksheet_data = [["hello", "world"]];
  var worksheet = XLSX.utils.aoa_to_sheet(worksheet_data);
  workbook.Sheets["Nodes"] = worksheet;

  var wbout = XLSX.write(workbook, { bookType: 'xlsx', type: 'binary' });

  //var URL = window.URL || window.webkitURL || window.mozURL || window.msURL || window.oURL;
  var blob = new Blob([binary_to_octet(wbout)], { type: "application/octet-stream" });

  saveAs(blob, 'test.xlsx');

  /*
    const fileBlob = new Blob(
      [binary_to_octet(wbout)],
      { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }
    );
    var objectURL = URL.createObjectURL(fileBlob);
  
    const exportLinkElement = document.createElement('a');
  
    exportLinkElement.hidden = true;
    exportLinkElement.download = "import_export.xlsx";
    exportLinkElement.href = objectURL;
    exportLinkElement.text = "downloading...";
  
    document.body.appendChild(exportLinkElement);
    exportLinkElement.click();
  
    URL.revokeObjectURL(objectURL);
    
    TODO:
    setTimeout(function() {
      URL.revokeObjectURL(objectURL);
    });
    
  
    exportLinkElement.remove();
    
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
    */
}