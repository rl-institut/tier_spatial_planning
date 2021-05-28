$(document).ready(function () {
  $(document).foundation();
  refreshNodeFromDataBase();
  refreshLinksFromDatBase();
});

// --------------------VARIABLES DECLARATION----------------------//

default_household_required_capacity = 10;
default_household_max_power = 20;

// --------------------FUNCTIONS DECLARATION----------------------//

// SET FUNCTIONS

function setMapClickEvent(mapClickEvent) {
  if (mapClickEvent === "node") {
    $(document.getElementById("button_draw_boundaries_add")).attr('disabled', true);
    $(document.getElementById("button_draw_boundaries_remove")).attr('disabled', true);
    $(document.getElementById("radio_button_node_high_demand")).attr('disabled', false);
    $(document.getElementById("radio_button_node_medium_demand")).attr('disabled', false);
    $(document.getElementById("radio_button_node_low_demand")).attr('disabled', false);
    $(document.getElementById("radio_button_node_pole")).attr('disabled', false);
  }
  else if (mapClickEvent === "boundary") {
    $(document.getElementById("button_draw_boundaries_add")).attr('disabled', false);
    $(document.getElementById("button_draw_boundaries_remove")).attr('disabled', false);
    $(document.getElementById("radio_button_node_high_demand")).attr('disabled', true);
    $(document.getElementById("radio_button_node_medium_demand")).attr('disabled', true);
    $(document.getElementById("radio_button_node_low_demand")).attr('disabled', true);
    $(document.getElementById("radio_button_node_pole")).attr('disabled', true);
  }
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
  $("#loading").show();
  $.ajax({
    url: "select_boundaries_add/",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      boundary_coordinates: boundariesCoordinates,
    }),
    dataType: "json",
    statusCode: {
      200: function () {
        refreshNodeFromDataBase();
        $("#loading").hide();
      },
    },
  });
}

function removeBuildingsInsideBoundary(boundariesCoordinates) {
  $("#loading").show();
  $.ajax({
    url: "select_boundaries_remove/",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      boundary_coordinates: boundariesCoordinates,
    }),
    dataType: "json",
    statusCode: {
      200: function () {
        refreshNodeFromDataBase();
        refreshLinksFromDatBase();
        $("#loading").hide();
      },
    },
  });
}

function addNodeToDatBase(
  latitude,
  longitude,
  area,
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
      area: area,
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
  $("#loading").show();
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
        $("#loading").hide();
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

  $("#loading").show();
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
        $("#loading").hide();
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
              <td>${node.area}</td>
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
        if (node.node_type === "high-demand") {
          markers.push(
            L.marker([node.latitude, node.longitude], {
              icon: markerHighDemand,
            }).addTo(mainMap)
          );
        } else if (node.node_type === "medium-demand") {
          markers.push(
            L.marker([node.latitude, node.longitude], {
              icon: markerMediumDemand,
            }).addTo(mainMap)
          );
        } else if (node.node_type === "low-demand") {
          markers.push(
            L.marker([node.latitude, node.longitude], {
              icon: markerLowDemand,
            }).addTo(mainMap)
          );
        } else if (node.node_type === "pole") {
          markers.push(
            L.marker([node.latitude, node.longitude], {
              icon: markerPole,
            }).addTo(mainMap)
          );
        } else {
          peak_demand_per_sq_meter = 4
          total_demand = node.area * peak_demand_per_sq_meter
          if (total_demand >= 100) {
            icon = markerHighDemand;
          } else if (total_demand < 100 && total_demand > 40) {
            icon = markerMediumDemand;
          } else {
            icon = markerLowDemand;
          }
          markers.push(
            L.marker([node.latitude, node.longitude],
              { icon: icon },
            ).addTo(mainMap)
          );
        }
      }
      if (mapClickEvent === "boundary") {
        zoomAll(mainMap);
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

// selecting boundaries of the site for adding new nodes
function selectBoundariesAdd() {
  mapClickEvent = "boundary";
  var textButtonDrawBoundariesAdd = document.getElementById(
    "button_draw_boundaries_add"
  );
  textButtonDrawBoundariesAdd.innerHTML = "Select";

  // changing the type of the button (primary <-> success)
  if ($(textButtonDrawBoundariesAdd).hasClass("primary")) {
    $(textButtonDrawBoundariesAdd).removeClass("primary");
    $(textButtonDrawBoundariesAdd).addClass("success");
  } else {
    $(textButtonDrawBoundariesAdd).removeClass("success");
    $(textButtonDrawBoundariesAdd).addClass("primary");
  }

  // adding a line to the list of lines inside the polyline object
  siteBoundaryLines.push(
    L.polyline([siteBoundaries[0], siteBoundaries.slice(-1)[0]], {
      color: "black",
    })
  );
  siteBoundaryLines[siteBoundaryLines.length - 1].addTo(mainMap);

  // TODO implement if close to check that area is not too large
  // TODO if clicking of Draw Lines, but without clicking on the map again choosing Select ...
  getBuildingCoordinates(siteBoundaries);

  removeBoundaries();

  textButtonDrawBoundariesAdd.innerHTML = "Draw Lines";

}

// selecting boundaries of the site for removing new nodes
function selectBoundariesRemove() {
  mapClickEvent = "boundary";
  var textButtonDrawBoundariesRemove = document.getElementById(
    "button_draw_boundaries_remove"
  );
  textButtonDrawBoundariesRemove.innerHTML = "Remove";

  // changing the type of the button (primary <-> alert)
  if ($(textButtonDrawBoundariesRemove).hasClass("primary")) {
    $(textButtonDrawBoundariesRemove).removeClass("primary");
    $(textButtonDrawBoundariesRemove).addClass("alert");
  } else {
    $(textButtonDrawBoundariesRemove).removeClass("alert");
    $(textButtonDrawBoundariesRemove).addClass("primary");
  }

  // adding a line to the list of lines inside the polyline object
  siteBoundaryLines.push(
    L.polyline([siteBoundaries[0], siteBoundaries.slice(-1)[0]], {
      color: "black",
    })
  );
  siteBoundaryLines[siteBoundaryLines.length - 1].addTo(mainMap);

  // TODO implement if close to check that area is not too large
  // TODO if clicking of Draw Lines, but without clicking on the map again choosing Select ...
  removeBuildingsInsideBoundary(siteBoundaries);

  removeBoundaries();

  textButtonDrawBoundariesRemove.innerHTML = "Draw Lines";

}