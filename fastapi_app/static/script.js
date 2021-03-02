
var mainMap = L.map('leafletMap').setView([9.07798, 7.704826], 5);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        tileSize: 512,
        zoomOffset: -1,
        minZoom: 1,
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        crossOrigin: true,
      }).addTo(mainMap);

var markers = [];
var lines = [];

mainMap.on('click', function(e) {
  var poplocation = e.latlng;

  addNodeToDatBase(poplocation.lat, poplocation.lng, "household", false)
  drawDefaultMarker(poplocation.lat, poplocation.lng)
});

function drawDefaultMarker(latitude, longitude) {
  markers.push(
    L.marker([latitude, longitude]).addTo(mainMap)
  );
}

function addNodeToDatBase(latitude, longitude, node_type, fixed_type) {
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

function optimize_grid(price_meterhub, price_household, price_interhub_cable, price_distribution_cable) {
  $.ajax({
    url: "optimize_grid",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      price_meterhub: price_meterhub,
      price_household: price_household,
      price_interhub_cable: price_interhub_cable,
      price_distribution_cable: price_distribution_cable,
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
        for (marker of markers){
          mainMap.removeLayer(marker);
        }
        markers.length = 0;
        for (node of nodes) {
          markers.push(L.marker([node.latitude, node.longitude]).addTo(mainMap))
        }
        
      }
    };
  }

function drawLinkOnMap(latitude_from,
                       longitude_from,
                       latitude_to,
                       longitude_to,
                       color,
                       map,
                       weight=3,
                       opacity=0.5) {
  var pointA = new L.LatLng(latitude_from, longitude_from);
  var pointB = new L.LatLng(latitude_to, longitude_to);
  var pointList = [pointA, pointB];

  var link_polyline = new L.Polyline(pointList, {
      color: color,
      weight: weight,
      opacity: 0.5,
      smoothFactor: 1
    });
  lines.push(
    link_polyline.addTo(map)
  );
}

function ereaseLinksFromMap(map) {
  for (line of lines){
    map.removeLayer(line);
  }
  lines.length = 0;
}

function refreshLinkTable() {
    var tbody_links = document.getElementById("tbody_links");
    var xhr = new XMLHttpRequest();
    url = "links_db_html";
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send();

    xhr.onreadystatechange = function () {
      if (this.readyState == 4 && this.status == 200) {
        links = this.response;
        html_link_table = "";
        for (link of links) {
          html_link_table += `
              <tr>
              <td>${link.id}</td>
              <td>${link.lat_from}</td>
              <td>${link.long_from}</td>
              <td>${link.lat_to}</td>
              <td>${link.long_to}</td>
              <td>${link.cable_type}</td>
              <td>${link.distance}</td>
              </tr>`;
            }
        tbody_links.innerHTML = html_link_table;
        ereaseLinksFromMap(mainMap)
        for (link of links) {
          var color = (link.cable_type === "interhub") ? "red":"green"
          drawLinkOnMap(link.lat_from,
                        link.long_from,
                        link.lat_to,
                        link.long_to,
                        color,
                        mainMap)
        }
        }
      }
    };
  



  $(document).ready(function () {
    refreshNodeTable();
    refreshLinkTable();

    setInterval(refreshNodeTable, 3000);
    setInterval(refreshLinkTable, 3000);
    

    $("#button_add_node").click(function () {
      const latitude = new_node_lat.value;
      const longitude = new_node_long.value;
      const node_type = new_node_type.value;
      const fixed_type = new_node_type_fixed.value;

      addNodeToDatBase(latitude, longitude, node_type, fixed_type)
      
    });
   
    $("#button_optimize").click(function () {
      const price_hub = hub_price.value;
      const price_household = household_price.value;
      const price_interhub_cable = interhub_cable_price.value;
      const price_distribution_cable = interhub_cable_price.value;
      optimize_grid(price_hub, price_household, price_interhub_cable, price_distribution_cable)
      
    });

    $("#button_clear_node_db").click(function () {
      $.ajax({
        url: "clear_node_db",
        type: "POST",
      });
    });
  });
