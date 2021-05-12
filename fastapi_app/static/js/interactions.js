$(document).ready(function () {
  refreshNodeFromDataBase();
  refreshLinksFromDatBase();
});

// --------------------VARIABLES DECLARATION----------------------//

default_household_required_capacity = 10;
default_household_max_power = 20;

// --------------------FUNCTIONS DECLARATION----------------------//

// SET FUNCTIONS

function setMapClickEventToAddNode() {
  mapClickEvent = "add_node";
}

function displayShsCharacteristicsInput() {
  if (document.getElementById("shs_inputs").style.display === "block") {
    document.getElementById("shs_inputs").style.display = "none";
  } else {
    document.getElementById("shs_inputs").style.display = "block";
  }
}

function logShsCharacteristics() {
  shsCharacteristics = [];
  for (var i = 0; i < 4; ++i) {
    shsCapacity = document.getElementById(`shs_capacity_${i}`).value;
    maxPower = document.getElementById(`shs_max_power_${i}`).value;
    price = document.getElementById(`shs_price_${i}`).value;

    if (shsCapacity > 0 && maxPower > 0 && price > 0) {
      shsCharacteristics.push({
        price: price,
        capacity: shsCapacity,
        max_power: maxPower,
      });
    }
  }
  return shsCharacteristics;
}

// POST REQUESTS

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
      refreshNodeFromDataBase();
    }
  };
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
    statusCode: {
      200: function () {
        refreshNodeFromDataBase();
      },
    },
  });
}

function optimize_grid() {
  $.ajax({
    url: "optimize_grid/",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      price_meterhub: price_meterhub.value,
      price_household: price_household.value,
      price_interhub_cable: price_interhub_cable.value,
      price_distribution_cable: price_distribution_cable.value,
      number_of_relaxation_steps_nr: number_of_relaxation_steps_nr.value,
    }),
    dataType: "json",
    statusCode: {
      200: function () {
        refreshNodeFromDataBase();
        refreshLinksFromDatBase();
      },
    },
  });
}

function identify_shs() {
  const cable_price_per_meter =
    cable_price_per_meter_for_shs_mst_identification.value;
  const additional_connection_price =
    additional_connection_price_for_shs_mst_identification.value;
  const algo = "mst1";
  const shs_characteristics = logShsCharacteristics();

  $.ajax({
    url: "shs_identification/",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      cable_price_per_meter_for_shs_mst_identification: cable_price_per_meter,
      additional_connection_price_for_shs_mst_identification:
        additional_connection_price,
      algo,
      shs_characteristics,
    }),
    dataType: "json",
    statusCode: {
      200: function () {
        refreshNodeFromDataBase();
        clearLinksDataBase();
      },
    },
  });
}

function refreshNodeFromDataBase() {
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

function refreshLinksFromDatBase() {
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

function clearLinksDataBase() {
  $.ajax({
    url: "clear_link_db/",
    type: "POST",
    statusCode: {
      200: function () {
        refreshLinksFromDatBase();
      },
    },
  });
}

function clear_node_db() {
  $.ajax({
    url: "clear_node_db/",
    type: "POST",
    statusCode: {
      200: function () {
        refreshNodeFromDataBase();
        refreshLinksFromDatBase();
      },
    },
  });
}

function validateBoundaries() {
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
}

function selectBoundaries() {
  mapClickEvent = "draw_boundaries";
  var textSelectBoundaryButton = document.getElementById(
    "button_select_boundaries"
  );
  if (textSelectBoundaryButton.innerHTML === "Reset") {
    mainMap.removeLayer(siteGeojson);
  }
  textSelectBoundaryButton.innerHTML = "Reset";

  removeBoundaries();
}
