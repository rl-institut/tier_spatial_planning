async function importConfig() {
  let formData = new FormData();
  formData.append("file", config_import.files[0]);

  $.ajax({
    url: "import_config",
    type: "POST",
    data: formData,
    processData: false,
    contentType: false,
    statusCode: {
      200: function () {
        refreshNodeFromDataBase();
        refreshLinksFromDatBase();
      },
    },
  });
}

function exportConfig() {
  window.open("export_config");
}
