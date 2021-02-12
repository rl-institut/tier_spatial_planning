
var mainMap = L.map('leafletMap').setView([51.505, -0.09], 2);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        tileSize: 512,
        zoomOffset: -1,
        minZoom: 1,
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        crossOrigin: true,
      }).addTo(mainMap);

mainMap.on('click', function(e) {
  var poplocation = e.latlng;
  console.log(poplocation)
  console.log("add node to database")
  addNodeToDatBase(e.latlng.lat, e.latlng.lng, "household", false)
  L.marker([e.latlng.lat, e.latlng.lng]).addTo(mainMap)
});

function addNodeToDatBase(latitude, longitude, node_type, fixed_type) {
  console.log("add node to database")
  
  $.ajax({
    url: "add_node",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      latitude: latitude,
      longitude: longitude,
      node_type: node_type,
      fixed_type: fixed_type,
    }),
    dataType: "json",
  });
}


function refreshNodeTable() {
    var tbody_nodes = document.getElementById("tbody_nodes");
    var xhr = new XMLHttpRequest();
    url = "nodes_db_html";
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send();

    xhr.onreadystatechange = function () {
      if (this.readyState == 4 && this.status == 200) {
        nodes = this.response;
        html_node_table = "";
        for (node of nodes) {
          html_node_table += `
              <tr>
              <td>${node.id}</td>
              <td>${node.latitude}</td>
              <td>${node.longitude}</td>
              <td>${node.node_type}</td>
              <td>${node.fixed_type}</td>
              </tr>`;
        }
        tbody_nodes.innerHTML = html_node_table;
      }
    };
  }

  $(document).ready(function () {
    refreshNodeTable();

    setInterval(refreshNodeTable, 1000);

    $("#button_add_node").click(function () {
      const latitude = new_node_lat.value;
      const longitude = new_node_long.value;
      const node_type = new_node_type.value;
      const fixed_type = new_node_type_fixed.value;

      addNodeToDatBase(latitude, longitude, node_type, fixed_type)
      
    });

    $("#button_clear_node_db").click(function () {
      $.ajax({
        url: "clear_node_db",
        type: "POST",
      });
    });
  });
