$(document).ready(function () {
    $(document).foundation();
    csv_files_initialization();
    //    refreshNodeFromDataBase();
    //    refreshLinksFromDatBase();
});

// --------------------VARIABLES DECLARATION----------------------//

default_household_required_capacity = 10;
default_household_max_power = 20;
// --------------------FUNCTIONS DECLARATION----------------------//

// SET FUNCTIONS

function setVisibilityNodeBox() {
    if (document.getElementById("radio_button_nodes_manually").checked) {
        $(document.getElementById("button_draw_boundaries_add")).attr(
            "disabled",
            true
        );
        $(document.getElementById("button_draw_boundaries_remove")).attr(
            "disabled",
            true
        );
        $(document.getElementById("radio_button_node_high_demand")).attr(
            "disabled",
            false
        );
        $(document.getElementById("radio_button_node_medium_demand")).attr(
            "disabled",
            false
        );
        $(document.getElementById("radio_button_node_low_demand")).attr(
            "disabled",
            false
        );
        $(document.getElementById("radio_button_node_pole")).attr(
            "disabled",
            false
        );
    } else if (document.getElementById("radio_button_nodes_boundaries").checked) {
        $(document.getElementById("button_draw_boundaries_add")).attr(
            "disabled",
            false
        );
        $(document.getElementById("button_draw_boundaries_remove")).attr(
            "disabled",
            false
        );
        $(document.getElementById("radio_button_node_high_demand")).attr(
            "disabled",
            true
        );
        $(document.getElementById("radio_button_node_medium_demand")).attr(
            "disabled",
            true
        );
        $(document.getElementById("radio_button_node_low_demand")).attr(
            "disabled",
            true
        );
        $(document.getElementById("radio_button_node_pole")).attr("disabled", true);
    }
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
    max_power,
    is_connected
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
            is_connected: is_connected,
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
            price_pole: price_pole.value,
            price_household: price_household.value,
            price_pole_cable: price_pole_cable.value,
            price_distribution_cable: price_distribution_cable.value,
            number_of_relaxation_steps_nr: number_of_relaxation_steps_nr.value,
            max_connection_poles: max_connection_poles.value,
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
    const max_distance_between_poles = 40; // must be definded globally in the fututre
    const cable_pole_price_per_meter =
        price_pole_cable.value + price_pole.value / max_distance_between_poles;
    const algo = "mst1";
    $("#loading").show();
    $.ajax({
        url: "shs_identification/",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({
            cable_price_per_meter_for_shs_mst_identification: cable_pole_price_per_meter,
            connection_cost_to_minigrid: price_household.value,
            price_shs_hd: price_shs_hd.value,
            price_shs_md: price_shs_md.value,
            price_shs_ld: price_shs_ld.value,
            algo,
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

/* getting properties of nodes and links from the stored data 
PARAMETERS:
    nodes: if equal to 'true' nodes will be considered
    links: if equal to 'true' links will be considered
*/
function csv_files_initialization() {
    var xhr = new XMLHttpRequest();
    url = "csv_files_initialization";
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send();
}


function csv_files_reading(read_nodes, read_links) {
    var xhr = new XMLHttpRequest();
    url = "reading_from_csv/" + read_nodes + "/" + read_links;
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send();

    xhr.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            nodes = this.response;
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
                } else if (node.node_type === "shs") {
                    markers.push(
                        L.marker([node.latitude, node.longitude], {
                            icon: markerShs,
                        }).addTo(mainMap)
                    );
                }
            }
            if (document.getElementById("radio_button_nodes_boundaries").checked) {
                zoomAll(mainMap);
            }
        }
    };
}


function db_add(
    {add_nodes=false,
    add_links=false,
    latitude=0,
    longitude=0,
    x=0,
    y=0,
    area=0,
    node_type,
    peak_demand=0,
    is_connected=true} = {}
) {
    $.ajax({
        url: "/db_add/" + add_nodes + "/" + add_links,
        type: "POST",
        dataType: "json",
        contentType: "application/json",
        data: JSON.stringify({
            latitude: latitude,
            longitude: longitude,
            x: x,
            y: y,
            area: area,
            node_type: node_type,
            peak_demand: peak_demand,
            is_connected: is_connected,
        }),
        statusCode: {
            200: function () {
                refreshNodeFromDataBase();
            },
        },
    });
}


function refreshNodeFromDataBase() {
    var xhr = new XMLHttpRequest();
    url = "nodes_db_html";
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send();

    xhr.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            nodes = this.response;
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
                } else if (node.node_type === "shs") {
                    markers.push(
                        L.marker([node.latitude, node.longitude], {
                            icon: markerShs,
                        }).addTo(mainMap)
                    );
                }
            }
            if (document.getElementById("radio_button_nodes_boundaries").checked) {
                zoomAll(mainMap);
            }
        }
    };
}

function refreshLinksFromDatBase() {
    var xhr = new XMLHttpRequest();
    url = "links_db_html";
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send();

    xhr.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            links = this.response;
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
    var textButtonDrawBoundariesAdd = document.getElementById(
        "button_draw_boundaries_add"
    );

    // changing the label of the button
    if (textButtonDrawBoundariesAdd.innerHTML === "Select") {
        textButtonDrawBoundariesAdd.innerHTML = "Draw Lines";
        textButtonDrawBoundariesAdd.setAttribute(
            "title",
            "Draw a polygon on the map to add nodes"
        );
    } else {
        textButtonDrawBoundariesAdd.innerHTML = "Select";
        textButtonDrawBoundariesAdd.setAttribute(
            "title",
            "Select all nodes inside the drawn polygon"
        );
    }

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

    // only when a boundary is drawn, the next steps will be executed
    if (siteBoundaryLines.length > 0) {
        getBuildingCoordinates(siteBoundaries);
        removeBoundaries();
        textButtonDrawBoundariesAdd.innerHTML = "Draw Lines";
    }
}

// selecting boundaries of the site for removing new nodes
function selectBoundariesRemove() {
    var textButtonDrawBoundariesRemove = document.getElementById(
        "button_draw_boundaries_remove"
    );

    // changing the label of the button
    if (textButtonDrawBoundariesRemove.innerHTML === "Remove") {
        textButtonDrawBoundariesRemove.innerHTML = "Draw Lines";
        textButtonDrawBoundariesRemove.setAttribute(
            "title",
            "Draw a polygon on the map to remove nodes"
        );
    } else {
        textButtonDrawBoundariesRemove.innerHTML = "Remove";
        textButtonDrawBoundariesRemove.setAttribute(
            "title",
            "Remove all nodes inside the drawn polygon"
        );
    }

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

    // only when a boundary is drawn, the next steps will be executed
    if (siteBoundaryLines.length > 0) {
        removeBuildingsInsideBoundary(siteBoundaries);
        removeBoundaries();
        textButtonDrawBoundariesRemove.innerHTML = "Draw Lines";
    }
}