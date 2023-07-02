

function db_links_to_js(project_id) {
  const url = "db_links_to_js/" + project_id;

  fetch(url)
    .then((response) => {
      if (response.ok) {
        return response.json();
      } else {
        throw new Error("Failed to fetch data");
      }
    })
    .then((links) => {
      // push links to the map
      removeLinksFromMap(map);
      for (let index = 0; index < Object.keys(links.link_type).length; index++) {
        var color = links.link_type[index] === "distribution" ? "rgb(255, 99, 71)" : "rgb(0, 165, 114)";
        var weight = links.link_type[index] === "distribution" ? 3 : 2;
        var opacity = links.link_type[index] === "distribution" ? 1 : 1;
        drawLinkOnMap(
          links.lat_from[index],
          links.lon_from[index],
          links.lat_to[index],
          links.lon_to[index],
          color,
          map,
          weight,
          opacity
        );
      }
    })
    .catch((error) => {
      console.error("Error fetching data:", error);
    });
}



async function db_nodes_to_js(project_id, markers_only) {
    fetch("/db_nodes_to_js/" + project_id + '/' + markers_only)
  .then(response => response.json())
  .then(data => {
    map_elements = data;
    if (map_elements !== null) {
        put_markers_on_map(map_elements, markers_only);
        }
    else {
        map_elements = [];
    }
  })
}

async function consumer_to_db(project_id) {
  update_map_elements();
  const url = "/consumer_to_db/" + project_id;
  await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({map_elements: map_elements})
  }).then(() => forward_if_consumer_selection_exists(project_id))
}

function add_buildings_inside_boundary({ boundariesCoordinates } = {}) {
  $("*").css("cursor", "wait");
  fetch("/add_buildings_inside_boundary", {
    method: "POST",
    headers: {"Content-Type": "application/json",},
    body: JSON.stringify({boundary_coordinates: boundariesCoordinates, map_elements: map_elements,}),
  })
    .then((response) => {
      if (response.ok) {
        return response.json();
      } else {
        throw new Error("Failed to fetch data");
      }
    })
    .then((res) => {
      $("*").css('cursor','auto');
      const responseMsg = document.getElementById("responseMsg");
      responseMsg.innerHTML = res.msg;
      if (res.executed === false) {
      } else {
        responseMsg.innerHTML = "";
        Array.prototype.push.apply(map_elements, res.new_consumers);
        put_markers_on_map(res.new_consumers, true);
      }
    })
    .catch((error) => {
      console.error("Error fetching data:", error);
    });
}

function remove_buildings_inside_boundary(
    {   boundariesCoordinates } = {},) {
    $("*").css("cursor", "wait");
    $.ajax({
        url: "/remove_buildings_inside_boundary",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({
            boundary_coordinates: boundariesCoordinates,
            map_elements: map_elements,
        }),
        dataType: "json",
    }).done(function (res) {
        $("*").css('cursor','auto');
        map_elements = res.map_elements;
        remove_marker_from_map();
        put_markers_on_map(map_elements, true)})}



/************************************************************/
/*                       OPTIMIZATION                       */
/************************************************************/


function save_energy_system_design() {
        $.ajax({
        url: "save_energy_system_design/",
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
                    'fuel_cost': dieselGensetFuelCost.value,
                    'fuel_lhv': dieselGensetFuelLhv.value,
                    'min_load': dieselGensetMinLoad.value/100,
                    'max_efficiency': dieselGensetMaxEfficiency.value/100,
                    'max_load': dieselGensetMaxLoad.value/100,
                    'min_efficiency': dieselGensetMinEfficiency.value/100,
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
                    'max_shortage_total': shortageMaxTotal.value/100,
                    'max_shortage_timestep': shortageMaxTimestep.value/100,
                    'shortage_penalty_cost': shortagePenaltyCost.value
                },
            },
        }),
        dataType: "json",
    }
    );
}


async function load_results(project_id) {
    var xhr = new XMLHttpRequest();
    url = "load_results/" + project_id;
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send();
    xhr.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            // push nodes to the map
            results = this.response;
            if (results['n_consumers'] > 0) {
            document.getElementById('noResults').style.display='none';
            document.getElementById("nConsumers").innerText = results['n_consumers'];
            document.getElementById("nShsConsumers").innerText = results['n_shs_consumers'];
            document.getElementById("nPoles").innerText = results['n_poles'];
            document.getElementById("maxVoltageDrop").innerText = results['max_voltage_drop'];
            document.getElementById("lengthDistributionCable").innerText = results['length_distribution_cable'];
            document.getElementById("averageLengthDistributionCable").innerText = results['average_length_distribution_cable'];
            document.getElementById("lengthConnectionCable").innerText = results['length_connection_cable'];
            document.getElementById("averageLengthConnectionCable").innerText = results['average_length_connection_cable'];
            document.getElementById("res").innerText = results['res'];
            document.getElementById("surplusRate").innerText = results['surplus_rate'];
            document.getElementById("shortageTotal").innerText = results['shortage_total'];
            document.getElementById("co2Savings").innerText = results['co2_savings'];
            document.getElementById("lcoe").innerText = results['lcoe'];
            document.getElementById("time").innerText = results['time'];
            db_nodes_to_js(project_id, false);
            db_links_to_js(project_id);
            plot();
        }
        else {
            document.getElementById('dashboard').style.display = 'none';
            document.getElementById('map').style.display = 'none';
            document.getElementById('noResults').style.display='block';
            $.ajax({
            url: "has_pending_task/" + project_id,
            type: "POST",
            contentType: "application/json",
            })
            .done(function (res) {
                if (res.has_pending_task === true)
                {
                    document.getElementById("pendingTaskMSG").innerText = 'Calculation is still running.';
                }
                else
                {
                    document.getElementById("pendingTaskMSG").innerText = 'There is no ongoing calculation.' +
                        ' If you want to run the optimization for this project, go to projects overview and click on ' +
                        '\"Edit Project\"';
                }
            });
    }}
}}


/************************************************************/
/*                    User Registration                     */
/************************************************************/


async function add_user_to_db()
{    $("*").css('cursor','wait');
     const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
     await sleep(3000); // Pause for 3 seconds (3000 milliseconds)

    if (userPassword2.value !== userPassword3.value)
    {
        document.getElementById("responseMsg2").innerHTML = 'The passwords do not match';
        document.getElementById("responseMsg2").style.color = 'red';
    }
    else
    {
    $.ajax({url: "add_user_to_db/",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({'email': userEmail2.value,
                                       'password': userPassword2.value,
                                         'captcha_input': captcha_input2.value,
                                       'hashed_captcha': hashedCaptcha}),
            dataType: "json",})
        .done(function (response) {
            document.getElementById("responseMsg2").innerHTML = response.msg;
            let fontcolor;
            if (response.validation === true)
                {fontcolor = 'green';}
            else
                {fontcolor = 'red';};
            document.getElementById("responseMsg2").style.color = fontcolor;});
    }
    $("*").css('cursor','auto');
}


async function change_email() {
    if (userEmail1.value !== userEmail2.value) {
        document.getElementById("responseMsg1").innerHTML = 'The emails do not match';
        document.getElementById("responseMsg1").style.color = 'red';
    }
    else {
    $("*").css("cursor", "wait");
    $.ajax({url: "change_email/",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({email: userEmail1.value,
                                        password: userPassword.value,
                                        remember_me: false}),
            dataType: "json",})
        .done(async function (response) {
            document.getElementById("responseMsg1").innerHTML = response.msg;
            let fontcolor;
            if (response.validation === true)
                {fontcolor = 'green';}
            else
                {fontcolor = 'red';};
            document.getElementById("responseMsg1").style.color = fontcolor;
            if (response.validation === true)
            {
            await new Promise(r => setTimeout(r, 3000))
            logout()
            }
        });
    $("*").css("cursor", "auto");
    }

}


async function change_pw() {
    if (newUserPassword1.value != newUserPassword2.value) {
        document.getElementById("responseMsg2").innerHTML = 'The passwords do not match';
        document.getElementById("responseMsg2").style.color = 'red';
    }
    else {
    $("*").css("cursor", "wait");
    $.ajax({url: "change_pw/",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({new_password: newUserPassword1.value,
                                        old_password: oldUserPassword.value}),
            dataType: "json",})
        .done(async function (response) {

            document.getElementById("responseMsg2").innerHTML = response.msg;
            let fontcolor;
            if (response.validation === true)
                {fontcolor = 'green';}
            else
                {fontcolor = 'red';};
            document.getElementById("responseMsg2").style.color = fontcolor;
            if (response.validation === true)
            {
            await new Promise(r => setTimeout(r, 3000))
            logout()
            }
        });
    $("*").css("cursor", "auto");
    }
}


function delete_account() {
    $("*").css("cursor", "wait");
    $.ajax({url: "delete_account/",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({password: Password.value}),
            dataType: "json",})
        .done(async function (response) {
            document.getElementById("responseMsg3").innerHTML = response.msg;
            let fontcolor;
            if (response.validation === true)
                {fontcolor = 'green';}
            else
                {fontcolor = 'red';};
            document.getElementById("responseMsg3").style.color = fontcolor;
            if (response.validation === true)
            {
            await new Promise(r => setTimeout(r, 3000))
            logout()
            }
        });
    $("*").css("cursor", "auto");
}


function login() {
    $.ajax({url: "login/",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({email: userEmail.value,
                                       password: userPassword.value, remember_me: isEnabled.value}),
            dataType: "json",})
        .done(function (response) {
            document.getElementById("userPassword").value = '';
            if (response.validation === true)
                {
                    document.getElementById("userEmail").value = '';
                    location.reload();
                }
            else
                {
                    document.getElementById("responseMsg").innerHTML = response.msg;
                    document.getElementById("responseMsg").style.color = 'red';
                }
            });}


async function renewToken() {
    const response = await fetch('/renew_token', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    });
    if (response.ok) {
        const data = await response.json();
        if (data && data.access_token) {
            localStorage.setItem('token', data.access_token);
        }
    }
}


function consent_cookie() {
    $.ajax({url: "consent_cookie/",
            type: "POST",
            contentType: "application/json",
            dataType: "json",})
        .done(function () {document.getElementById('consentCookie').style.display='none'});}


function anonymous_login () {
    $.ajax({url: "anonymous_login/",
            type: "POST",
            data: JSON.stringify({'captcha_input': captcha_input3.value,
                                  'hashed_captcha': hashedCaptcha}),
            contentType: "application/json",
            dataType: "json",})
        .done(async function (response) {
            document.getElementById("responseMsg3").innerHTML = response.msg;
            if (response.validation === true)
            {window.location.href=window.location.origin;}
            else
            {
             document.getElementById("responseMsg3").style.color = 'red';
            }
        });
}


function logout()  {
    $.ajax({url: "logout/",
            type: "POST",
            contentType: "application/json",
            dataType: "json"})
        .done(function () {window.location.href=window.location.origin;});

}


function save_project_setup(project_id) {
    $.ajax({
        url: "save_project_setup/" + project_id,
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify(
            {
                page_setup: {
                    'project_name': projectName.value,
                    'project_description': projectDescription.value.trim(),
                    'interest_rate': interestRate.value,
                    'project_lifetime': projectLifetime.value,
                    'start_date': startDate.value,
                    'temporal_resolution': temporalResolution.value,
                    'n_days': nDays.value,
                }
            }),
        dataType: "json",});}


function save_grid_design() {
    fetch("save_grid_design/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            grid_design: {
                'distribution_cable_lifetime': distributionCableLifetime.value,
                'distribution_cable_capex': distributionCableCapex.value,
                'distribution_cable_max_length': distributionCableMaxLength.value,
                'connection_cable_lifetime': connectionCableLifetime.value,
                'connection_cable_capex': connectionCableCapex.value,
                'connection_cable_max_length': connectionCableMaxLength.value,
                'pole_lifetime': poleLifetime.value,
                'pole_capex': poleCapex.value,
                'pole_max_n_connections': poleMaxNumberOfConnections.value,
                'mg_connection_cost': mgConnectionCost.value,
                'shs_max_grid_cost': shs_max_grid_cost.value,
            }
        })
    }).then(response => response.json());
}

function save_demand_estimation() {
    let custom_calibration = document.getElementById("toggleswitch").checked;
    fetch("save_demand_estimation/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            demand_estimation: {
                'household_option': selectedValue,
                'maximum_peak_load': maximum_peak_load.value,
                'average_daily_energy': average_daily_energy.value,
                'custom_calibration': custom_calibration,
            }
        })
    }).then(response => response.json());
}


function load_previous_data(page_name){
    var xhr = new XMLHttpRequest();
    url = "load_previous_data/" + page_name;
    xhr.open("GET", url, true);
    xhr.responseType = "json";
    xhr.send();
    if (page_name.includes("project_setup")) {
        xhr.onreadystatechange = function () {

            if (this.readyState == 4 && this.status == 200) {
                // push nodes to the map
                results = this.response;
                if (results !== null && Object.keys(results).length > 1){
                    document.getElementById("projectName").value = results['project_name'];
                    document.getElementById("projectDescription").value = results['project_description'];
                    document.getElementById("interestRate").value = results['interest_rate'];
                    document.getElementById("projectLifetime").value = results['project_lifetime'];
                    document.getElementById("startDate").value = results['start_date'];
                    document.getElementById("temporalResolution").value = results['temporal_resolution'];
                    document.getElementById("nDays").value = results['n_days'];
                }
            }
        };
    } else if (page_name.includes("grid_design")) {
        xhr.onreadystatechange = function () {
            if (this.readyState == 4 && this.status == 200) {
                // push nodes to the map
                results = this.response;
                if (results !== null && Object.keys(results).length > 1) {
                    document.getElementById("distributionCableLifetime").value = results['distribution_cable_lifetime'];
                    document.getElementById("distributionCableCapex").value = results['distribution_cable_capex'];
                    document.getElementById("distributionCableMaxLength").value = results['distribution_cable_max_length'];
                    document.getElementById("connectionCableLifetime").value = results['connection_cable_lifetime'];
                    document.getElementById("connectionCableCapex").value = results['connection_cable_capex'];
                    document.getElementById("connectionCableMaxLength").value = results['connection_cable_max_length'];
                    document.getElementById("poleLifetime").value = results['pole_lifetime'];
                    document.getElementById("poleCapex").value = results['pole_capex'];
                    document.getElementById("poleMaxNumberOfConnections").value = results['pole_max_n_connections'];
                    document.getElementById("mgConnectionCost").value = results['mg_connection_cost'];
                    document.getElementById("shs_max_grid_cost").value = results['shs_max_grid_cost'];
                }
            }
        };
    } else if (page_name.includes("demand_estimation")) {
        xhr.onreadystatechange = function () {
            if (this.readyState == 4 && this.status == 200) {
                // push nodes to the map
                results = this.response;
                if (results !== null && Object.keys(results).length > 1) {
                    document.getElementById("maximum_peak_load").value = results['maximum_peak_load'];
                    document.getElementById("average_daily_energy").value = results['average_daily_energy'];
                    document.getElementById("toggleswitch").checked = results['custom_calibration'];
                    let accordionItem2 = new bootstrap.Collapse(document.getElementById('collapseTwo'),
                        {toggle: false});
                    if (results['custom_calibration'] == true){
                        accordionItem2.show();}
                    else {
                        accordionItem2.hide();
                    }
                  const radioButton = document.querySelector(`input[name="options"][id="option${results['household_option'] + 1}"]`);

                  if (radioButton) {
                    radioButton.checked = true;
                  }
                    }
                }
            }
        }
    else if (page_name.includes("energy_system_design")) {
        xhr.onreadystatechange = function () {
            if (this.readyState == 4 && this.status == 200) {
                // push nodes to the map
                results = this.response;
                if (results !== null && Object.keys(results).length > 1) {
                    document.getElementById("selectPv").checked = results['pv__settings__is_selected'];
                    document.getElementById("pvDesign").checked = results['pv__settings__design'];
                    document.getElementById("pvNominalCapacity").value = results['pv__parameters__nominal_capacity'];
                    document.getElementById("pvLifetime").value = results['pv__parameters__lifetime'];
                    document.getElementById("pvCapex").value = results['pv__parameters__capex'];
                    document.getElementById("pvOpex").value = results['pv__parameters__opex'];
                    document.getElementById("selectDieselGenset").checked = results['diesel_genset__settings__is_selected'];
                    document.getElementById("dieselGensetDesign").checked = results['diesel_genset__settings__design'];
                    document.getElementById("dieselGensetCapex").value = results['diesel_genset__parameters__capex'];
                    document.getElementById("dieselGensetOpex").value = results['diesel_genset__parameters__opex'];
                    document.getElementById("dieselGensetVariableCost").value = results['diesel_genset__parameters__variable_cost'];
                    document.getElementById("dieselGensetFuelCost").value = results['diesel_genset__parameters__fuel_cost'];
                    document.getElementById("dieselGensetFuelLhv").value = results['diesel_genset__parameters__fuel_lhv'];
                    document.getElementById("dieselGensetMinLoad").value = results['diesel_genset__parameters__min_load'] * 100;
                    document.getElementById("dieselGensetMaxEfficiency").value = results['diesel_genset__parameters__max_efficiency'] * 100;
                    document.getElementById("dieselGensetMaxLoad").value = results['diesel_genset__parameters__max_load'] * 100;
                    document.getElementById("dieselGensetMinEfficiency").value = results['diesel_genset__parameters__min_efficiency'] * 100;
                    document.getElementById("dieselGensetLifetime").value = results['diesel_genset__parameters__lifetime'];
                    document.getElementById("dieselGensetNominalCapacity").value = results['diesel_genset__parameters__nominal_capacity'];
                    document.getElementById("selectInverter").checked = results['inverter__settings__is_selected'];
                    document.getElementById("inverterDesign").checked = results['inverter__settings__design'];
                    document.getElementById("inverterNominalCapacity").value = results['inverter__parameters__nominal_capacity'];
                    document.getElementById("inverterLifetime").value = results['inverter__parameters__lifetime'];
                    document.getElementById("inverterCapex").value = results['inverter__parameters__capex'];
                    document.getElementById("inverterOpex").value = results['inverter__parameters__opex'];
                    document.getElementById("inverterEfficiency").value = results['inverter__parameters__efficiency'] * 100;
                    document.getElementById("selectRectifier").checked = results['rectifier__settings__is_selected'];
                    document.getElementById("rectifierDesign").checked = results['rectifier__settings__design'];
                    document.getElementById("rectifierNominalCapacity").value = results['rectifier__parameters__nominal_capacity'];
                    document.getElementById("rectifierLifetime").value = results['rectifier__parameters__lifetime'];
                    document.getElementById("rectifierCapex").value = results['rectifier__parameters__capex'];
                    document.getElementById("rectifierOpex").value = results['rectifier__parameters__opex'];
                    document.getElementById("rectifierEfficiency").value = results['rectifier__parameters__efficiency'] * 100;
                    document.getElementById("selectShortage").checked = results['shortage__settings__is_selected'];
                    document.getElementById("shortageMaxTotal").value = results['shortage__parameters__max_shortage_total'] * 100;
                    document.getElementById("shortageMaxTimestep").value = results['shortage__parameters__max_shortage_timestep'] * 100;
                    document.getElementById("shortagePenaltyCost").value = results['shortage__parameters__shortage_penalty_cost'];
                    document.getElementById("selectBattery").checked = results['battery__settings__is_selected'];
                    document.getElementById("batteryDesign").checked = results['battery__settings__design'];
                    document.getElementById("batteryNominalCapacity").value = results['battery__parameters__nominal_capacity'];
                    document.getElementById("batteryLifetime").value = results['battery__parameters__lifetime'];
                    document.getElementById("batteryCrateIn").value = results['battery__parameters__c_rate_in'];
                    document.getElementById("batteryCrateOut").value = results['battery__parameters__c_rate_out'];
                    document.getElementById("batterySocMin").value = results['battery__parameters__soc_min'] * 100;
                    document.getElementById("batterySocMax").value = results['battery__parameters__soc_max'] * 100;
                    document.getElementById("batteryEfficiency").value = results['battery__parameters__efficiency'] * 100;
                    document.getElementById("batteryOpex").value = results['battery__parameters__opex'];
                    document.getElementById("batteryCapex").value = results['battery__parameters__capex'];
                    document.getElementById("batteryNominalCapacity").value = results['battery__parameters__nominal_capacity'];
                    }
                }
            }
        }
}


/************************************************************/
/*                   EXPORT DATA AS XLSX                    */
/************************************************************/

function export_data(project_id) {
    // Create the excel workbook and fill it out with some properties
    var workbook = XLSX.utils.book_new();s
    workbook.Props = {
      Title: "Import and Export Data form/to the Optimization Web App.",
      Subject: "Off-Grid Network and Energy Supply System",
      Author: "Saeed Sayadi",
      CreatedDate: new Date(2022, 8, 8)
    };
  
    // Get all nodes from the database.
    db_links_to_js(nodes_or_links = 'nodes', map_or_export = 'export', project_id, function (data_nodes) {
        
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

function import_data(project_id) {
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
            url: "import_data/" + project_id,
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({
            nodes_to_import,
            links_to_import
            }),
            dataType: "json",
            statusCode: {
            200: function () {
                db_links_to_js(nodes_or_links = 'nodes', map_or_export = 'map', project_id);
                db_links_to_js(nodes_or_links = 'links', map_or_export = 'map', project_id);
            },
            },
        });
    }
  }



function show_user_email_in_navbar() {
    fetch("query_account_data/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
    })
        .then(response => response.json())
        .then(data => {
            const showMailElement = document.getElementById("showMail");
            if (showMailElement) {
                showMailElement.innerHTML = data.email;
            }
        });
}



async function redirect_if_cookie_is_missing(access_token, consent_cookie){
        let has_access_token = (access_token === true || access_token === 'true');
        let has_consent_cookie = (consent_cookie === true || consent_cookie === 'true');
        $.ajax({url: "has_cookie/",
            type: "POST",
            data: JSON.stringify({'access_token': has_access_token, 'consent_cookie': has_consent_cookie}),
            contentType: "application/json",})
        .done(function (response) {
            if (response == false)
            {   logout();
                window.location.href = window.location.origin;}
        })
        await renewToken();
}


function remove_project(project_id) {
        $.ajax({url: "remove_project/" + project_id,
            type: "POST",
            contentType: "application/json",})
        .done(function () {window.location.href = window.location.origin;})}


function wait_for_results(project_id, task_id, time, model)
{   $.ajax({
        url: "waiting_for_results/",
        type: "POST",
        data: JSON.stringify({'project_id': project_id, 'task_id': task_id, 'time': time, 'model': model}),
        contentType: "application/json",
    })
    .done(function (res) {
        if (res.finished === true) {
            window.location.href = window.location.origin + '/simulation_results?project_id=' + project_id;
        } else {
            document.querySelector("#statusMsg").innerHTML = res.status;
            renewToken();
            wait_for_results(project_id, task_id, res.time, res.model);
        }
    });
}


function forward_if_no_task_is_pending(project_id) {
        $.ajax({
            url: "forward_if_no_task_is_pending/",
            type: "POST",
            contentType: "application/json",})
        .done(function (res) {
        if (res.forward === true) {
            window.location.href = window.location.origin + '/calculating?project_id=' + project_id;
        } else {
            document.getElementById('pendingTask').style.display='block'
        }})}


function revoke_users_task() {
    $.ajax({
        url: "revoke_users_task/",
        type: "POST",
        contentType: "application/json"})
        .done(function () {document.getElementById('pendingTask').style.display='none';
        })}


function start_calculation(project_id)
    {   $.ajax({
            url: "start_calculation/" + project_id,
            type: "POST",
            contentType: "application/json",
        })
        .done(function (res) {
            wait_for_results(project_id, res.task_id, 0, 'grid');
        });
    }

function forward_if_consumer_selection_exists(project_id) {
    $.ajax({
        url: "forward_if_consumer_selection_exists/" + project_id,
        type: "POST",
        contentType: "application/json",})
            .done(function (res) {
        if (res.forward === true) {
            window.location.href = window.location.origin + '/demand_estimation?project_id=' + project_id;
        } else {
            document.getElementById('responseMsg').innerHTML = 'No consumers are selected. You must select the geolocation of the consumers before you go\n' +
                '                    to the next page.';
        }})}



function send_email_notification(project_id, is_active) {
    $.ajax({
        url: "/set_email_notification/" + project_id + '/' + is_active,
        type: "POST",
        contentType: "application/json",
    });
}

function show_cookie_consent(){
        $.ajax({url: "has_cookie/",
            type: "POST",
            data: JSON.stringify({'access_token': false, 'consent_cookie': true}),
            contentType: "application/json",})
        .done(function (response) {
            if (response == false)
            {
                document.getElementById('consentCookie').style.display='block'}
            else
            {document.getElementById('consentCookie').style.display='none'}
        })
}


function send_reset_password_email(){
        $.ajax({url: "send_reset_password_email/",
            type: "POST",
            data: JSON.stringify({'email': userEmail4.value,
                                        'captcha_input': captcha_input.value,
                                        'hashed_captcha': hashedCaptcha}),
            contentType: "application/json",})
        .done(async function (response) {
            document.getElementById("responseMsg4").innerHTML = response.msg;
            let fontcolor;
            if (response.validation === true)
                {fontcolor = 'green';}
            else
                {fontcolor = 'red';};
            document.getElementById("responseMsg4").style.color = fontcolor;
            if (response.validation === true)
            {
            await new Promise(r => setTimeout(r, 3000))
            document.getElementById('forgotPassword').style.display='none'
            }
        });
}

function reset_pw(guid) {
    if (newUserPassword1.value !== newUserPassword2.value) {
        document.getElementById("responseMsg2").innerHTML = 'The passwords do not match';
        document.getElementById("responseMsg2").style.color = 'red';
    }
    else {
    $.ajax({url: "reset_password",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({guid: guid, password: newUserPassword1.value}),
            dataType: "json",})
        .done(async function (response) {

            document.getElementById("responseMsg2").innerHTML = response.msg;
            let fontcolor;
            if (response.validation === true)
                {fontcolor = 'green';}
            else
                {fontcolor = 'red';};
            document.getElementById("responseMsg2").style.color = fontcolor;
            if (response.validation === true)
            {
            await new Promise(r => setTimeout(r, 3000))
            window.location.href=window.location.origin;
            }
        });
    }
}
