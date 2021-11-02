$(document).ready(function () {
    $(document).foundation();
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
// note: both "nodes" and "links" cannot be called simultaneously
function database_to_map(nodes_or_links) {
    var xhr = new XMLHttpRequest();
    url = "database_to_map/" + nodes_or_links;
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send();

    xhr.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
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
                        if (nodes["demand_type"][counter] === "high-demand") {
                            markers.push(
                                L.marker([nodes["latitude"][counter], nodes["longitude"][counter]], {
                                    icon: markerHighDemand,
                                }).addTo(mainMap)
                            );
                        } else if (nodes["demand_type"][counter] === "medium-demand") {
                            markers.push(
                                L.marker([nodes["latitude"][counter], nodes["longitude"][counter]], {
                                    icon: markerMediumDemand,
                                }).addTo(mainMap)
                            );
                        } else if (nodes["demand_type"][counter] === "low-demand") {
                            markers.push(
                                L.marker([nodes["latitude"][counter], nodes["longitude"][counter]], {
                                    icon: markerLowDemand,
                                }).addTo(mainMap)
                            );
                        }
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
                        links.long_from[index],
                        links.lat_to[index],
                        links.long_to[index],
                        color,
                        mainMap,
                        weight
                    );
                }
            }
        }
    };
}


// add single nodes selected manually to the *.csv file
function database_add_manual(
    { latitude,
        longitude,
        area = 0,
        node_type,
        consumer_type,
        demand_type,
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
            area: area,
            node_type: node_type,
            consumer_type: consumer_type,
            demand_type: demand_type,
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
                database_to_map(nodes_or_links = 'nodes');
                $("#loading").hide();
            },
        },
    });
}


/************************************************************/
/*                    BOUNDARY SELECTION                    */
/************************************************************/

// selecting boundaries of the site for adding new nodes
function boundary_select(mode) {
    if (mode == 'add') {
        button_text = 'Select'
        button_class = 'success'
        var textButtonDrawBoundaries = document.getElementById(
            "button_draw_boundaries_add"
        );
    } else {
        var textButtonDrawBoundaries = document.getElementById(
            "button_draw_boundaries_remove"
        );
        button_text = 'Remove'
        button_class = 'alert'
    }

    // changing the label of the button
    if (textButtonDrawBoundaries.innerHTML === button_text) {
        textButtonDrawBoundaries.innerHTML = "Draw Lines";
        textButtonDrawBoundaries.setAttribute(
            "title",
            "Draw a polygon on the map to" + mode + "nodes"
        );
    } else {
        textButtonDrawBoundaries.innerHTML = button_text;
        textButtonDrawBoundaries.setAttribute(
            "title",
            button_text + "all nodes inside the drawn polygon"
        );
    }

    // changing the type of the button (primary <-> success)
    if ($(textButtonDrawBoundaries).hasClass("primary")) {
        $(textButtonDrawBoundaries).removeClass("primary");
        $(textButtonDrawBoundaries).addClass(button_class);
    } else {
        $(textButtonDrawBoundaries).removeClass(button_class);
        $(textButtonDrawBoundaries).addClass("primary");
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
        textButtonDrawBoundaries.innerHTML = "Draw Lines";
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
                database_to_map(nodes_or_link = 'nodes');
                database_to_map(nodes_or_link = 'links');
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
                database_to_map(nodes_or_links = 'nodes');
                //refreshNodeFromDataBase();
                clearLinksDataBase();
                $("#loading").hide();
            },
        },
    });
}