// $(document).ready(function () {
//     // $(document).foundation();
//     database_read(nodes_or_links = 'nodes', map_or_export = 'map');

//     // database_initialization(nodes = true, links = true);
// });


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
                    zoomAll(mainMap);
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
        url: "/database_add_remove_automatic/" + add_remove,
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
// function customer_selection() {
//     $.ajax({
//         url: "/customer_selection",
//         type: "POST",
//         dataType: "json",
//         contentType: "application/json",
//         data: JSON.stringify({
            
//         }),
//     });
// }

/************************************************************/
/*                ACTIVATION / INACTIVATION                 */
/************************************************************/
function activation_check() {
    if (document.getElementById('selectionFile').checked) {
        document.getElementById('fileImport').disabled = false
        document.getElementById('btnImport').classList.remove('disabled');
        document.getElementById('lblDrawBoundariesAdd').classList.add('disabled');
        document.getElementById('btnDrawBoundariesAdd').classList.add('disabled');
        document.getElementById('lblDrawBoundariesRemove').classList.add('disabled');
        document.getElementById('btnDrawBoundariesRemove').classList.add('disabled');
    } else if (document.getElementById('selectionBoundaries').checked) {
        document.getElementById('fileImport').disabled = true
        document.getElementById('btnImport').classList.add('disabled');
        document.getElementById('lblDrawBoundariesAdd').classList.remove('disabled');
        document.getElementById('btnDrawBoundariesAdd').classList.remove('disabled');
        document.getElementById('lblDrawBoundariesRemove').classList.remove('disabled');
        document.getElementById('btnDrawBoundariesRemove').classList.remove('disabled');
    } else if (document.getElementById('selectionMap').checked) {
        document.getElementById('fileImport').disabled = true
        document.getElementById('btnImport').classList.add('disabled');
        document.getElementById('lblDrawBoundariesAdd').classList.add('disabled');
        document.getElementById('btnDrawBoundariesAdd').classList.add('disabled');
        document.getElementById('lblDrawBoundariesRemove').classList.add('disabled');
        document.getElementById('btnDrawBoundariesRemove').classList.add('disabled');
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
        btnAddRemove.innerText = 'Draw Lines';
        btnAddRemove.classList.toggle(button_class)
    } else {
        btnAddRemove.innerText = button_text;
        btnAddRemove.classList.remove(button_class)
    }

    // add a line to the polyline object in the map
    siteBoundaryLines.push(
        L.polyline([siteBoundaries[0], siteBoundaries.slice(-1)[0]], {
            color: 'black',
        })
    );
    siteBoundaryLines[siteBoundaryLines.length - 1].addTo(mainMap);

    // only when a boundary is drawn, the next steps will be executed
    if (siteBoundaryLines.length > 0) {
        database_add_remove_automatic({ add_remove: mode, boundariesCoordinates: siteBoundaries });
        removeBoundaries();
    }
}


/************************************************************/
/*                       OPTIMIZATION                       */
/************************************************************/

function optimize_energy_system() {
    // $("#loading").show();
    $.ajax({
        url: "optimize_energy_system/",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({
            start_date: localStorage.start_date,
            n_days: (localStorage.n_days),
            project_lifetime: parseInt(localStorage.project_lifetime),
            wacc: parseFloat(localStorage.interest_rate)/100,
            tax: 0,
            pv: {'lifetime': pvLifetime.value, 'capex': pvCapex.value, 'opex': pvOpex.value},
            diesel_genset: {'lifetime': dieselGensetLifetime.value, 'capex': dieselGensetCapex.value, 
                            'opex': dieselGensetOpex.value, 'variable_cost': dieselGensetVariableCost.value,
                            'fuel_cost': fuelCost.value, 'fuel_lhv': fuelLhv.value, 
                            'min_load': dieselGensetMinLoad.value/100, 'efficiency': dieselGensetEfficiency.value/100},
            battery: {'lifetime': batteryLifetime.value, 'capex': batteryCapex.value, 'opex': batteryOpex.value,
                      'soc_min': batterySocMin.value/100, 'soc_max': batterySocMax.value/100, 'c_rate_in': batteryCrateIn.value, 
                      'c_rate_out': batteryCrateOut.value, 'efficiency': batteryEfficiency.value/100},
            inverter: {'lifetime': inverterLifetime.value, 'capex': inverterCapex.value, 'opex': inverterOpex.value, 'efficiency': inverterEfficiency.value/100},
            rectifier: {'lifetime': rectifierLifetime.value, 'capex': rectifierCapex.value, 'opex': rectifierOpex.value, 'efficiency': rectifierEfficiency.value/100},
                }),
        dataType: "json",
    });
}

function optimize_grid() {
    document.getElementById('spnSpinner').style.display = "";
    
    $.ajax({
        url: "optimize_grid/",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({
            start_date: localStorage.start_date,
            n_days: (localStorage.n_days),
            project_lifetime: parseInt(localStorage.project_lifetime),
            wacc: parseFloat(localStorage.interest_rate)/100,
            tax: 0,
            hv_cable: {'lifetime': hvCableLifetime.value, 'capex': hvCableCapex.value, 'opex': hvCableOpex.value},
            hv_cable: {'lifetime': hvCableLifetime.value, 'capex': hvCableCapex.value, 'opex': hvCableOpex.value},
            lv_cable: {'lifetime': lvCableLifetime.value, 'capex': lvCableCapex.value, 'opex': lvCableOpex.value},
            pole: {'lifetime': poleLifetime.value, 'capex': poleCapex.value, 'opex': poleOpex.value, 'max_connections': 4},
            connection: {'lifetime': connectionLifetime.value, 'capex': connectionCapex.value, 'opex': connectionOpex.value},
            optimization: {'n_relaxation_steps': 10},
                }),
        dataType: "json",
    });

    // window.open("{{ url_for('simulation_results')}}");
}

function load_results(){
    var xhr = new XMLHttpRequest();
    url = "load_results/";
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send();
    
    xhr.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            // push nodes to the map
            results = this.response;
            document.getElementById("nPoles").innerText = results['n_poles'];
            document.getElementById("nConsumers").innerText = results['n_consumers'];
            document.getElementById("lengthHvCable").innerText = results['length_hv_cable'];
            document.getElementById("lengthLvCable").innerText = results['length_lv_cable'];
            document.getElementById("costGrid").innerText = results['cost_grid'];
        }
    };
}

function refresh_map(){
    database_read(nodes_or_link = 'nodes', map_or_export = 'map');
    database_read(nodes_or_link = 'links', map_or_export = 'map');
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

function project_setup_variables(){
    localStorage.start_date = document.getElementById("startDate").value;
    localStorage.n_days = document.getElementById("nDays").value;
    localStorage.project_lifetime = document.getElementById("projectLifetime").value;
    localStorage.interest_rate = document.getElementById("interestRate").value;
}