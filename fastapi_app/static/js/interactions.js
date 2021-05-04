default_household_required_capacity = 10;
default_household_max_power = 20;

var osmLayer = L.tileLayer(
  "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  {
    tileSize: 512,
    zoomOffset: -1,
    minZoom: 1,
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    crossOrigin: true,
  }
);

var osmMap = {
  osmBaseMap: osmLayer,
};

var esriWorldImageryLayer = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  {
    tileSize: 512,
    zoomOffset: -1,
    minZoom: 1,
    maxZoom: 18,
    attribution:
      "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
  }
);

var esriSatelliteMap = {
  esriBaseMap: esriWorldImageryLayer,
};

var mainMap = L.map("leafletMap", {
  center: [11.3929, 9.1248],
  zoom: 17,
  layers: [osmLayer],
});

L.control.layers(osmMap, esriSatelliteMap).addTo(mainMap);

var mapClickEvent = "add_node";

function setMapClickEventToAddNode() {
  mapClickEvent = "add_node";
}

var siteBoundaries = [];

var siteBoundaryLines = [];
var dashedBoundaryLine = null;

siteGeojson = "";

L.control.scale().addTo(mainMap);

var householdMarker = new L.Icon({
  iconUrl: "static/images/markers/marker-household.png",
  iconSize: [10, 10],
  iconAnchor: [5, 5],
  popupAnchor: [0, 0],
});

var hubMarker = new L.Icon({
  iconUrl: "/static/images/markers/marker-hub.png",
  iconSize: [14, 14],
  iconAnchor: [7, 7],
  popupAnchor: [0, 0],
});

var shsMarker = new L.Icon({
  iconUrl: "static/images/markers/marker-shs.png",
  iconSize: [10, 10],
  iconAnchor: [5, 5],
  popupAnchor: [0, 0],
});

var markers = [];
var lines = [];

mainMap.on("click", function (e) {
  var poplocation = e.latlng;

  if (mapClickEvent == "add_node") {
    if (document.getElementsByName("radio_button_new_node_type")[0].checked) {
      addNodeToDatBase(
        poplocation.lat,
        poplocation.lng,
        "undefinded",
        false,
        default_household_required_capacity,
        default_household_max_power
      );
      drawDefaultMarker(poplocation.lat, poplocation.lng);
    }

    if (document.getElementsByName("radio_button_new_node_type")[1].checked) {
      addNodeToDatBase(
        poplocation.lat,
        poplocation.lng,
        "household",
        true,
        default_household_required_capacity,
        default_household_max_power
      );
      drawHouseholdMarker(poplocation.lat, poplocation.lng);
    }

    if (document.getElementsByName("radio_button_new_node_type")[2].checked) {
      addNodeToDatBase(
        poplocation.lat,
        poplocation.lng,
        "meterhub",
        true,
        2 * default_household_required_capacity,
        2 * default_household_max_power
      );
      drawMeterhubMarker(poplocation.lat, poplocation.lng);
    }
  }

  if (mapClickEvent == "draw_boundaries") {
    siteBoundaries.push([poplocation.lat, poplocation.lng]);

    // add new solid line to siteBoundaryLines and draw it on map
    siteBoundaryLines.push(L.polyline(siteBoundaries, { color: "black" }));

    siteBoundaryLines[siteBoundaryLines.length - 1].addTo(mainMap);
    // Remove dashed line
    if (dashedBoundaryLine) {
      mainMap.removeLayer(dashedBoundaryLine);
    }

    // Create new dashed line closing the polygon
    dashedBoundaryLine = L.polyline(
      [siteBoundaries[0], siteBoundaries.slice(-1)[0]],
      { color: "black", dashArray: "10, 10", dashOffset: "20" }
    );

    // Add new dashed line to map
    dashedBoundaryLine.addTo(mainMap);
  }
});

function getBuildingCoordinates(boundariesCoordinates) {
  var xhr = new XMLHttpRequest();
  url = "/validate_boundaries";
  xhr.open("POST", url, true);
  xhr.setRequestHeader("Content-Type", "application/json");

  xhr.send(
    JSON.stringify({
      boundary_coordinates: boundariesCoordinates,
      default_required_capacity: default_household_required_capacity,
      default_max_power: default_household_max_power,
    })
  );
  xhr.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      siteGeojson = L.geoJSON(JSON.parse(xhr.responseText));

      siteGeojson.addTo(mainMap);
    }
  };
}

function drawDefaultMarker(latitude, longitude) {
  markers.push(L.marker([latitude, longitude]).addTo(mainMap));
}

function drawMeterhubMarker(latitude, longitude) {
  markers.push(
    L.marker([latitude, longitude], { icon: hubMarker }).addTo(mainMap)
  );
}

function drawHouseholdMarker(latitude, longitude) {
  markers.push(
    L.marker([latitude, longitude], { icon: householdMarker }).addTo(mainMap)
  );
}

function addNodeToDatBase(
  latitude,
  longitude,
  node_type,
  fixed_type,
  required_capacity,
  max_power
) {
  $.ajax({
    url: "add_node/",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      latitude: latitude,
      longitude: longitude,
      node_type: node_type,
      fixed_type: fixed_type,
      required_capacity: required_capacity,
      max_power: max_power,
    }),
    dataType: "json",
  });
}

function removeBoundaries() {
  // Remove all boundary lines and markers
  for (line of siteBoundaryLines) {
    mainMap.removeLayer(line);
  }
  if (dashedBoundaryLine != null) {
    mainMap.removeLayer(dashedBoundaryLine);
  }
  siteBoundaries.length = 0;
  siteBoundaryLines.length = 0;
  dashedBoundaryLine = null;
}

function optimize_grid(
  price_meterhub,
  price_household,
  price_interhub_cable,
  price_distribution_cable
) {
  $.ajax({
    url: "optimize_grid/",
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

function identify_shs(
  cable_price_per_meter,
  additional_connection_price,
  algo,
  shs_characteristics
) {
  console.log(
    JSON.stringify({
      cable_price_per_meter_for_shs_mst_identification: cable_price_per_meter,
      additional_connection_price_for_shs_mst_identification: additional_connection_price,
      algo,
      shs_characteristics,
    })
  );

  $.ajax({
    url: "shs_identification/",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      cable_price_per_meter_for_shs_mst_identification: cable_price_per_meter,
      additional_connection_price_for_shs_mst_identification: additional_connection_price,
      algo,
      shs_characteristics,
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
              <td>${node.required_capacity}</td>
              <td>${node.max_power}</td>
              </tr>`;
      }
      tbody_nodes.innerHTML = html_node_table;
      for (marker of markers) {
        mainMap.removeLayer(marker);
      }
      markers.length = 0;
      for (node of nodes) {
        if (node.node_type === "meterhub") {
          markers.push(
            L.marker([node.latitude, node.longitude], {
              icon: hubMarker,
            }).addTo(mainMap)
          );
        } else if (node.node_type === "household") {
          markers.push(
            L.marker([node.latitude, node.longitude], {
              icon: householdMarker,
            }).addTo(mainMap)
          );
        } else if (node.node_type === "shs") {
          markers.push(
            L.marker([node.latitude, node.longitude], {
              icon: shsMarker,
            }).addTo(mainMap)
          );
        } else {
          markers.push(
            L.marker([node.latitude, node.longitude]).addTo(mainMap)
          );
        }
      }
    }
  };
}

function drawLinkOnMap(
  latitude_from,
  longitude_from,
  latitude_to,
  longitude_to,
  color,
  map,
  weight = 3,
  opacity = 0.5
) {
  var pointA = new L.LatLng(latitude_from, longitude_from);
  var pointB = new L.LatLng(latitude_to, longitude_to);
  var pointList = [pointA, pointB];

  var link_polyline = new L.polyline(pointList, {
    color: color,
    weight: weight,
    opacity: 0.5,
    smoothFactor: 1,
  });
  lines.push(link_polyline.addTo(map));
}

function ereaseLinksFromMap(map) {
  for (line of lines) {
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
      ereaseLinksFromMap(mainMap);
      for (link of links) {
        var color = link.cable_type === "interhub" ? "red" : "green";
        drawLinkOnMap(
          link.lat_from,
          link.long_from,
          link.lat_to,
          link.long_to,
          color,
          mainMap
        );
      }
    }
  };
}

$(document).ready(function () {
  refreshNodeTable();
  refreshLinkTable();

  setInterval(refreshNodeTable, 3000);
  setInterval(refreshLinkTable, 3000);

  $("#button_add_undefined_node").click(function () {
    mapClickEvent = "add_node";
  });

  $("#button_add_household").click(function () {
    mapClickEvent = "add_fixed_household";
  });

  $("#button_add_meterhub").click(function () {
    mapClickEvent = "add_fixed_meterhub";
  });

  $("#button_add_node").click(function () {
    const latitude = new_node_lat.value;
    const longitude = new_node_long.value;
    const node_type = new_node_type.value;
    const fixed_type = new_node_type_fixed.value;
    const required_capacity = default_household_required_capacity;
    const max_power = default_household_max_power;

    addNodeToDatBase(
      latitude,
      longitude,
      node_type,
      fixed_type,
      required_capacity,
      max_power
    );
  });

  $("#button_optimize").click(function () {
    const price_hub = hub_price.value;
    const price_household = household_price.value;
    const price_interhub_cable = interhub_cable_price.value;
    const price_distribution_cable = distribution_cable_price.value;
    optimize_grid(
      price_hub,
      price_household,
      price_interhub_cable,
      price_distribution_cable
    );
  });

  $("#button_identify_shs").click(function () {
    const cable_price_per_meter =
      cable_price_per_meter_for_shs_mst_identification.value;
    const additional_connection_price =
      additional_connection_price_for_shs_mst_identification.value;
    const algo = "mst1";
    const shs_characteristics = logShsCharacteristics();

    identify_shs(
      cable_price_per_meter,
      additional_connection_price,
      algo,
      shs_characteristics
    );
  });

  $("#button_clear_node_db").click(function () {
    $.ajax({
      url: "clear_node_db/",
      type: "POST",
    });
  });

  $("#button_select_boundaries").click(function () {
    mapClickEvent = "draw_boundaries";
    var textSelectBoundaryButton = document.getElementById(
      "button_select_boundaries"
    );
    if (textSelectBoundaryButton.innerHTML === "Reset") {
      mainMap.removeLayer(siteGeojson);
    }
    textSelectBoundaryButton.innerHTML = "Reset";

    removeBoundaries();
  });

  $("#button_validate_boundaries").click(function () {
    mapClickEvent = "select";

    // Close polygone by changing dashed line to solid
    if (dashedBoundaryLine != null) {
      mainMap.removeLayer(dashedBoundaryLine);
    }
    siteBoundaryLines.push(
      L.polyline([siteBoundaries[0], siteBoundaries.slice(-1)[0]], {
        color: "black",
      })
    );
    siteBoundaryLines[siteBoundaryLines.length - 1].addTo(mainMap);

    // Find most extreme latitudes and longitudes
    const latitudeList = siteBoundaries.map((x) => x[0]);
    const longitudeList = siteBoundaries.map((x) => x[1]);

    minLatitude = Math.min(...latitudeList);
    maxLatitude = Math.max(...latitudeList);

    minLongitude = Math.min(...longitudeList);
    maxLongitude = Math.max(...longitudeList);

    // TODO implement if close to check that area is not too large

    getBuildingCoordinates((boundariesCoordinates = siteBoundaries));
  });
});

function logShsCharacteristics() {
  shsCharacteristics = [];
  for (var i = 0; i < 5; ++i) {
    shsCapacity = document.getElementById(`shs_capacity_${i}`).value;
    maxPower = document.getElementById(`shs_max_power_${i}`).value;
    price = document.getElementById(`shs_price_${i}`).value;

    if (shsCapacity > 0 && maxPower > 0 && price > 0) {
      shsCharacteristics.push({
        capacity: shsCapacity,
        max_power: maxPower,
        price: price,
      });
    }
  }
  return shsCharacteristics;
}
