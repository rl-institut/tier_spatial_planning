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
                    // As soon as there are nodes in the database, the download option will be activated
                    // otherwise, it will be disables. This only happens in the `customer-selection` page.
                    if (document.getElementById('btnDownloadLocations')) {
                        if (number_of_nodes > 0) {
                            document.getElementById('btnDownloadLocations').classList.remove('disabled');
                            document.getElementById('lblDownloadLocations').classList.remove('disabled');
                        } else {
                            document.getElementById('btnDownloadLocations').classList.add('disabled');
                            document.getElementById('lblDownloadLocations').classList.add('disabled');
                        }
                    };
                    var counter;
                    for (counter = 0; counter < number_of_nodes; counter++) {
                        if (nodes["node_type"][counter] === "pole") {
                            markers.push(
                                L.marker([nodes["latitude"][counter], nodes["longitude"][counter]], {
                                    icon: markerPole,
                                }).on('click', markerOnClick).addTo(mainMap)
                            );
                        } else if (nodes["is_connected"][counter] === false) {
                            // if the node is not connected to the grid, it will be a SHS consumer
                            markers.push(
                                L.marker([nodes["latitude"][counter], nodes["longitude"][counter]], {
                                    icon: markerShs,
                                }).on('click', markerOnClick).addTo(mainMap)
                            );
                        } else {
                            markers.push(
                                L.marker([nodes["latitude"][counter], nodes["longitude"][counter]], {
                                    icon: markerConsumer,
                                }).on('click', markerOnClick).addTo(mainMap)
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
function database_add_remove_manual(
    { add_remove = "add",
        latitude,
        longitude,
        node_type = 'consumer',
        consumer_type = 'household',
        consumer_detail = 'default',
        surface_area = 0,
        peak_demand = 0,
        average_consumption = 0,
        is_connected = true,
        how_added = 'manual' } = {}
) {
    $.ajax({
        url: "/database_add_remove_manual/" + add_remove,
        type: "POST",
        dataType: "json",
        contentType: "application/json",
        data: JSON.stringify({
            add_remove: add_remove,
            latitude: latitude,
            longitude: longitude,
            node_type: node_type,
            consumer_type: consumer_type,
            consumer_detail: consumer_detail,
            surface_area: surface_area,
            peak_demand: peak_demand,
            average_consumption: average_consumption,
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
            pv: {
                'settings': {
                    'is_selected': selectPv.checked, 
                     'design': pvDesign.checked, 
                },
                 'parameters': {
                    'nominal_capacity': pvNominalCapacity.value, 
                    'lifetime': pvLifetime.value, 
                    'capex': pvCapex.value, 
                    'opex': pvOpex.value,
                }
            },
            diesel_genset: {
                'settings': {
                    'is_selected': selectDieselGenset.checked, 
                     'design': dieselGensetDesign.checked, 
                },
                'parameters': {
                    'nominal_capacity': dieselGensetNominalCapacity.value,
                    'lifetime': dieselGensetLifetime.value, 
                    'capex': dieselGensetCapex.value, 
                    'opex': dieselGensetOpex.value, 
                    'variable_cost': dieselGensetVariableCost.value,
                    'fuel_cost': fuelCost.value, 
                    'fuel_lhv': fuelLhv.value, 
                    'min_load': dieselGensetMinLoad.value/100, 
                    'max_efficiency': dieselGensetMaxEfficiency.value/100,
                }
            },
            battery: {
                'settings': {
                    'is_selected': selectBattery.checked, 
                     'design': batteryDesign.checked, 
                },
                'parameters':{
                    'nominal_capacity': batteryNominalCapacity.value,
                    'lifetime': batteryLifetime.value, 'capex': batteryCapex.value, 'opex': batteryOpex.value,
                    'soc_min': batterySocMin.value/100, 'soc_max': batterySocMax.value/100, 'c_rate_in': batteryCrateIn.value, 
                    'c_rate_out': batteryCrateOut.value, 'efficiency': batteryEfficiency.value/100,
                } 
            },
            inverter: {
                'settings': {
                    'is_selected': selectInverter.checked, 
                    'design': inverterDesign.checked, 
                },
                'parameters': {
                    'nominal_capacity': inverterNominalCapacity.value,
                    'lifetime': inverterLifetime.value, 
                    'capex': inverterCapex.value, 
                    'opex': inverterOpex.value, 
                    'efficiency': inverterEfficiency.value/100,
                },
            },
            rectifier: {
                'settings': {
                    'is_selected': selectRectifier.checked, 
                    'design': rectifierDesign.checked, 
                },
                'parameters': {
                    'nominal_capacity': rectifierNominalCapacity.value,
                    'lifetime': rectifierLifetime.value, 
                    'capex': rectifierCapex.value, 
                    'opex': rectifierOpex.value, 
                    'efficiency': rectifierEfficiency.value/100
                },
            },
            shortage: {
                'settings': {
                    'is_selected': selectShortage.checked,
                },
                'parameters': {
                    'max_shortage_total': maxShortageTotal.value/100,
                    'max_shortage_timestep': maxShortageTimestep.value/100,
                    'shortage_penalty_cost': shortagePenaltyCost.value
                },
            },
        }),
        dataType: "json",
    });
}

// TODO: start date, interest rate, lifetime and wacc that come from another page are not recognized. 
// Either global parameters must be defined or something else.
function optimize_grid() {
    document.getElementById('spnSpinner').style.display = "";
    
    $.ajax({
        url: "optimize_grid/",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({
            optimization: {'n_relaxation_steps': 10},
            constraints: {'pole_max_connections': 4},
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
            document.getElementById("lcoe").innerText = results['lcoe'];
            document.getElementById("res").innerText = results['res'];
            document.getElementById("co2Savings").innerText = results['co2_savings'];
            document.getElementById("surplusRate").innerText = results['surplus_rate'];
            document.getElementById("solver").innerText = results['solver'];
            document.getElementById("gridOptimization").innerText = results['grid_optimization'];
            document.getElementById("time").innerText = results['time'];
        }
    };
}

function refresh_map(){
    database_read(nodes_or_link = 'nodes', map_or_export = 'map');
    database_read(nodes_or_link = 'links', map_or_export = 'map');

}

function save_previous_data(page_name) {
    if (page_name === "project_setup") {
        transfer_data = JSON.stringify(
            {
                page_setup: {
                    'project_name': projectName.value,
                    'project_description': projectDescription.value.trim(),
                    'interest_rate': interestRate.value,
                    'project_lifetime': projectLifetime.value,
                    'start_date': startDate.value,
                    'temporal_resolution': temporalResolution.value,
                    'n_days': nDays.value,
                },
                customer_selection: {
                    'hv_cable_lifetime': 0,
                    'hv_cable_capex': 0,
                    'lv_cable_lifetime': 0,
                    'lv_cable_capex': 0,
                    'pole_lifetime': 0,
                    'pole_capex': 0,
                    'mg_connection_cost': 0,
                    'shs_capex': 0,
                }
            }
        );
    } else if (page_name === "customer_selection") {
        transfer_data = JSON.stringify(
            {
                page_setup: {
                    'project_name': '',
                    'project_description': '',
                    'interest_rate': '',
                    'project_lifetime': '',
                    'start_date': '',
                    'temporal_resolution': '',
                    'n_days': '',
                },
                customer_selection: {
                    'hv_cable_lifetime': hvCableLifetime.value,
                    'hv_cable_capex': hvCableCapex.value,
                    'lv_cable_lifetime': lvCableLifetime.value,
                    'lv_cable_capex': lvCableCapex.value,
                    'pole_lifetime': poleLifetime.value,
                    'pole_capex': poleCapex.value,
                    'mg_connection_cost': mgConnectionCost.value,
                    'shs_capex': shsCapex.value,
                }
            }
        );
    }
    $.ajax({
        url: "save_previous_data/" + page_name,
        type: "POST",
        contentType: "application/json",
        data: transfer_data,
        dataType: "json",
    });
}

function load_previous_data(page_name){
    var xhr = new XMLHttpRequest();
    url = "load_previous_data/" + page_name;
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send();
    
    if (page_name === "project_setup") {
        xhr.onreadystatechange = function () {
            if (this.readyState == 4 && this.status == 200) {
                // push nodes to the map
                results = this.response;
                document.getElementById("projectName").value = results['project_name'];
                document.getElementById("projectDescription").value = results['project_description'];
                document.getElementById("interestRate").value = results['interest_rate'];
                document.getElementById("projectLifetime").value = results['project_lifetime'];
                document.getElementById("startDate").value = results['start_date'];
                document.getElementById("temporalResolution").value = results['temporal_resolution'];
                document.getElementById("nDays").value = results['n_days'];
            }
        };
    } else if (page_name === "customer_selection") {
        xhr.onreadystatechange = function () {
            if (this.readyState == 4 && this.status == 200) {
                // push nodes to the map
                results = this.response;
                document.getElementById("hvCableLifetime").value = results['hv_cable_lifetime'];
                document.getElementById("hvCableCapex").value = results['hv_cable_capex'];
                document.getElementById("lvCableLifetime").value = results['lv_cable_lifetime'];
                document.getElementById("lvCableCapex").value = results['lv_cable_capex'];
                document.getElementById("poleLifetime").value = results['pole_lifetime'];
                document.getElementById("poleCapex").value = results['pole_capex'];
                document.getElementById("mgConnectionCost").value = results['mg_connection_cost'];
                document.getElementById("shsCapex").value = results['shs_capex'];
            }
        };
    }
}

/************************************************************/
/*                   EXPORT DATA AS XLSX                    */
/************************************************************/

function export_data() {
    // Create the excel workbook and fill it out with some properties
    var workbook = XLSX.utils.book_new();
    workbook.Props = {
      Title: "Import and Export Data form/to the Optimization Web App.",
      Subject: "Off-Grid Network and Energy Supply System",
      Author: "Saeed Sayadi",
      CreatedDate: new Date(2022, 08, 08)
    };
  
    // Get all nodes from the database.
    database_read(nodes_or_links = 'nodes', map_or_export = 'export', function (data_nodes) {
        
        // Since the format of the JSON file exported by the `database_read` is
        // not compatible with the `Sheetjs` library, we need to restructure it
        // first. For this purpose, we require an array consisting of the same
        // number of elements as the number of nodes (representing the rows) and
        // for each element we should write down all properties in a dictionaty.

        // Here, all the properties of the nodes are read from the `data_nodes`
        // e.g., latitude, longitude, ...
        var headers = Object.keys(data_nodes);

        // To obtain the number of nodes, we take the first property of the
        // node, which can be any parameter depending on the `data_nodes`, and
        // obtain its length.
        var number_of_nodes = Object.keys(data_nodes[Object.keys(data_nodes)[0]]).length

        // The final JSON file must be like [{}, {}, {}, ...], which means there
        // are several numbers of single dictionaries (i.e., for each node),
        // and just one array that includes all these dictionaries.
        // This array is then put into the Excel file.
        var single_dict = {};
        var array_of_dicts = [];

        for(var item = 0; item < number_of_nodes; item++) {
            for(var header in headers) {
                single_dict[headers[header]] = data_nodes[headers[header]][item];
            }
            array_of_dicts.push(single_dict);
            // Remove the content of the `single_dict` to avoid using the same
            // numbers for different elements of the array.
            single_dict = {};
        }
        
        // Create an Excel worksheet from the the array consisting several
        // single dictionaries.
        const worksheet = XLSX.utils.json_to_sheet(array_of_dicts);

        // Create a new sheet in the Excel workbook with the given name and
        // copy the content of the `worksheet` into it.
        XLSX.utils.book_append_sheet(workbook, worksheet, 'Nodes')           

        // Specify the proper write options.
        let wopts = { bookType: 'xlsx', type: 'array'};

        // Get the current date and time with this format YYYY_M_D_H_M_S to add
        // to the end of the Excel file.
        const current_date = new Date(Date.now());
        const time_extension = current_date.getFullYear() + '_' +  (current_date.getMonth() + 1) + '_' +  current_date.getDate() + '_' + current_date.getHours() + '_' + current_date.getMinutes() + '_' + current_date.getSeconds();
        
        XLSX.writeFile(workbook, 'import_export_' + time_extension + '.xlsx', wopts);
    });
}


/************************************************************/
/*                   IMPORT DATA AS XLSX                    */
/************************************************************/

function import_data() {
    // Choose the selected file in the web app.
    var selected_file = document.getElementById("fileImport").files[0];
    let file_reader = new FileReader();
    file_reader.readAsBinaryString(selected_file);
    
    // In case that the file can be loaded without any problem, this will be 
    // executed.
    file_reader.onload = function (event) {
        let import_data = event.target.result;
        let workbook = XLSX.read(import_data, { type: "binary" });

        // TODO: must be finalized later
        // // import settings to the web app
        // let settings_row_object = XLSX.utils.sheet_to_row_object_array(workbook.Sheets['settings'], {
        //     blankrows: false,
        //     header: 1,
        //     raw: true,
        //     rawNumbers: true
        // });
        // settings_row_object.shift(); // remove the first array only containing the sheet name ("settings") and "value"
        // settings_dict = settings_row_object.reduce((dict, [key, value]) => Object.assign(dict, { [key]: value }), {}); // convert the array to dictionary
        // import_settings_to_webapp(settings_dict);

        // copy nodes and links into the existing *.csv files (Databases)
        let nodes_to_import = XLSX.utils.sheet_to_row_object_array(workbook.Sheets['Nodes']);
        let links_to_import = XLSX.utils.sheet_to_row_object_array(workbook.Sheets['Links']);
        $.ajax({
            url: "/import_data",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({
            nodes_to_import,
            links_to_import
            }),
            dataType: "json",
            statusCode: {
            200: function () {
                database_read(nodes_or_links = 'nodes', map_or_export = 'map');
                database_read(nodes_or_links = 'links', map_or_export = 'map');
            },
            },
        });
    }
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