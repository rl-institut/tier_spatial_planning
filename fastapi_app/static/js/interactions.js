$(document).ready(function () {
    // $(document).foundation();
    database_initialization(nodes = true, links = true);
});

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

/************************************************************/
/*                         DATABASE                         */
/************************************************************/

// remove the existing nodes and links in the *.csv files
function database_initialization(nodes, links) {
    var xhr = new XMLHttpRequest();
    url = "database_initialization/" + nodes + "/" + links;
    xhr.open("GET", url, true);
    xhr.send()
}

// read all nodes/links stored in the *.csv files
// then push the corresponding icon to the map
// or return their correcponding json files for exporting the excel file
// note: both "nodes" and "links" cannot be called simultaneously
function database_read(nodes_or_links, map_or_export, callback) {
    var xhr = new XMLHttpRequest();
    url = "database_to_js/" + nodes_or_links;
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send();

    xhr.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            if (map_or_export == 'export') {
                if (callback) callback(this.response);
            }
            else {
                if (nodes_or_links == 'nodes') {
                    // push nodes to the map
                    nodes = this.response;
                    for (marker of markers) {
                        mainMap.removeLayer(marker);
                    }
                    markers.length = 0;
                    number_of_nodes = Object.keys(nodes["node_type"]).length;
                    var counter;
                    for (counter = 0; counter < number_of_nodes; counter++) {
                        if (nodes["node_type"][counter] === "pole") {
                            markers.push(
                                L.marker([nodes["latitude"][counter], nodes["longitude"][counter]], {
                                    icon: markerPole,
                                }).addTo(mainMap)
                            );
                        } else if (nodes["is_connected"][counter] === false) {
                            // if the node is not connected to the grid, it will be a SHS consumer
                            markers.push(
                                L.marker([nodes["latitude"][counter], nodes["longitude"][counter]], {
                                    icon: markerShs,
                                }).addTo(mainMap)
                            );
                        } else {
                            markers.push(
                                L.marker([nodes["latitude"][counter], nodes["longitude"][counter]], {
                                    icon: markerConsumer,
                                }).addTo(mainMap)
                            );
                        }
                    }
                    if (document.getElementById("radio_button_nodes_boundaries").checked) {
                        zoomAll(mainMap);
                    }
                } else {
                    // push links to the map
                    links = this.response;
                    removeLinksFromMap(mainMap);
                    for (let index = 0; index < Object.keys(links.link_type).length; index++) {
                        var color = links.link_type[index] === "interpole" ? "red" : "green";
                        var weight = links.link_type[index] === "interpole" ? 5 : 3;
                        drawLinkOnMap(
                            links.lat_from[index],
                            links.lon_from[index],
                            links.lat_to[index],
                            links.lon_to[index],
                            color,
                            mainMap,
                            weight
                        );
                    }
                }
            }
        }
    };
}


// add single nodes selected manually to the *.csv file
function database_add_manual(
    { latitude,
        longitude,
        node_type = 'consumer',
        consumer_type = 'household',
        consumer_detail = 'default',
        average_consumption = 0,
        peak_demand = 0,
        is_connected = true,
        how_added = 'manual' } = {}
) {
    $.ajax({
        url: "/database_add_manual",
        type: "POST",
        dataType: "json",
        contentType: "application/json",
        data: JSON.stringify({
            latitude: latitude,
            longitude: longitude,
            node_type: node_type,
            consumer_type: consumer_type,
            consumer_detail: consumer_detail,
            average_consumption: average_consumption,
            peak_demand: peak_demand,
            is_connected: is_connected,
            how_added: how_added,
        }),
    });
}


// add/remove nodes automatically from a given boundary
function database_add_remove_automatic(
    { add_remove = "add",
        boundariesCoordinates } = {}
) {
    $("#loading").show();
    $.ajax({
        url: "database_add_remove_automatic/" + add_remove,
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({
            boundary_coordinates: boundariesCoordinates,
        }),
        dataType: "json",
        statusCode: {
            200: function () {
                database_read(nodes_or_links = 'nodes', map_or_export = 'map');
                $("#loading").hide();
            },
        },
    });
}

/************************************************************/
/*                   SWITCHING HTML PAGES                   */
/************************************************************/
function customer_selection() {
    var xhr = new XMLHttpRequest();
    url = "customer_selection/";
    xhr.open("GET", url, true);
    xhr.send()
}

/************************************************************/
/*                ACTIVATION / INACTIVATION                 */
/************************************************************/
function activation_check() {
    if (document.getElementById('selectionFile').checked) {
        document.getElementById('fileImport').disabled = false
        document.getElementById('btnImport').classList.remove('disabled');
        document.getElementById('lblDrawBoundariesAdd').classList.toggle('disabled');
        document.getElementById('btnDrawBoundariesAdd').classList.toggle('disabled');
        document.getElementById('lblDrawBoundariesRemove').classList.toggle('disabled');
        document.getElementById('btnDrawBoundariesRemove').classList.toggle('disabled');
        document.getElementById('lblDownloadLocations').classList.remove('disabled');
        document.getElementById('btnDownloadLocations').classList.remove('disabled');
    } else if (document.getElementById('selectionBoundaries').checked) {
        document.getElementById('fileImport').disabled = true
        document.getElementById('btnImport').classList.toggle('disabled');
        document.getElementById('lblDrawBoundariesAdd').classList.remove('disabled');
        document.getElementById('btnDrawBoundariesAdd').classList.remove('disabled');
        document.getElementById('lblDrawBoundariesRemove').classList.remove('disabled');
        document.getElementById('btnDrawBoundariesRemove').classList.remove('disabled');
        document.getElementById('lblDownloadLocations').classList.remove('disabled');
        document.getElementById('btnDownloadLocations').classList.remove('disabled');
    } else if (document.getElementById('selectionMap').checked) {
        document.getElementById('fileImport').disabled = true
        document.getElementById('btnImport').classList.toggle('disabled');
        document.getElementById('lblDrawBoundariesAdd').classList.toggle('disabled');
        document.getElementById('btnDrawBoundariesAdd').classList.toggle('disabled');
        document.getElementById('lblDrawBoundariesRemove').classList.toggle('disabled');
        document.getElementById('btnDrawBoundariesRemove').classList.toggle('disabled');
        document.getElementById('lblDownloadLocations').classList.remove('disabled');
        document.getElementById('btnDownloadLocations').classList.remove('disabled');
    }
}
/************************************************************/
/*                    BOUNDARY SELECTION                    */
/************************************************************/
// selecting boundaries of the site for adding new nodes
function boundary_select(mode) {
    button_text = 'Start'
    if (mode == 'add') {
        button_class = 'btn--success'
        var btnAddRemove = document.getElementById("btnDrawBoundariesAdd");
    } else {
        var btnAddRemove = document.getElementById("btnDrawBoundariesRemove");
        button_class = 'btn--error'
    }

    // changing the label of the button
    if (btnAddRemove.innerText === button_text) {
        btnAddRemove.innerText = "Draw Lines";
        btnAddRemove.classList.toggle(button_class)
    } else {
        btnAddRemove.innerText = button_text;
        btnAddRemove.classList.remove(button_class)
    }

    // add a line to the polyline object in the map
    siteBoundaryLines.push(
        L.polyline([siteBoundaries[0], siteBoundaries.slice(-1)[0]], {
            color: "black",
        })
    );
    siteBoundaryLines[siteBoundaryLines.length - 1].addTo(mainMap);

    // only when a boundary is drawn, the next steps will be executed
    if (siteBoundaryLines.length > 0) {
        database_add_remove_automatic({ add_remove: mode, boundariesCoordinates: siteBoundaries });
        removeBoundaries();
        btnAddRemove.innerText = "Draw Lines";
    }
}


/************************************************************/
/*                       OPTIMIZATION                       */
/************************************************************/

function optimize_grid() {
    $("#loading").show();
    $.ajax({
        url: "optimize_grid/",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({
            cost_pole: cost_pole.value,
            cost_connection: cost_connection.value,
            cost_interpole_cable: cost_interpole_cable.value,
            cost_distribution_cable: cost_distribution_cable.value,
            number_of_relaxation_steps_nr: number_of_relaxation_steps_nr.value,
            max_connection_poles: max_connection_poles.value,
        }),
        dataType: "json",
        statusCode: {
            200: function () {
                database_read(nodes_or_link = 'nodes', map_or_export = 'map');
                database_read(nodes_or_link = 'links', map_or_export = 'map');
                $("#loading").hide();
            },
        },
    });
}

/************************************************************/
/*                    SOLAR-HOME-SYSTEM                     */
/************************************************************/

function identify_shs() {
    const max_distance_between_poles = 40; // must be definded globally in the fututre
    const cable_pole_price_per_meter =
        cost_interpole_cable.value + cost_pole.value / max_distance_between_poles;
    const algo = "mst1";
    $("#loading").show();
    $.ajax({
        url: "shs_identification/",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({
            cable_price_per_meter_for_shs_mst_identification: cable_pole_price_per_meter,
            connection_cost_to_minigrid: cost_connection.value,
            price_shs_hd: price_shs_hd.value,
            price_shs_md: price_shs_md.value,
            price_shs_ld: price_shs_ld.value,
            algo,
        }),
        dataType: "json",
        statusCode: {
            200: function () {
                database_read(nodes_or_links = 'nodes', map_or_export = 'map');
                //refreshNodeFromDataBase();
                clearLinksDataBase();
                $("#loading").hide();
            },
        },
    });
}