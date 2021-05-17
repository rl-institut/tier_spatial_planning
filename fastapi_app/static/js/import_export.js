async function uploadConfig() {
  //   let formData = new FormData();
  //   formData.append("file", config_upload.files[0]);

  //   $.ajax({
  //     url: "upload_config/",
  //     type: "POST",
  //     data: formData,
  //     statusCode: {
  //       200: function () {
  //         refreshNodeFromDataBase();
  //       },
  //     },
  //   });

  var file = config_upload.files[0];
  var reader = new FileReader();
  reader.onload = function () {
    $.ajax({
      url: "import_config/",
      type: "POST",
      data: reader.result,
    });
  };
  reader.readAsBinaryString(file);
}
