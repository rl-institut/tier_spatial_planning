function plot() {
    const urlParams = new URLSearchParams(window.location.search);
    project_id = urlParams.get('project_id');
    fetch('/get_plot_data/' + project_id + '/demand_coverage')
        .then(response => response.json())
        .then(data => {
            plot_demand_coverage(data.demand_coverage);
        });
    fetch('/get_plot_data/' + project_id + '/energy_flow')
        .then(response => response.json())
        .then(data => {
            plot_energy_flows(data.energy_flow);
        });
    fetch('/get_plot_data/' + project_id + '/other')
        .then(response => response.json())
        .then(data => {
            plot_lcoe_pie(data.lcoe_breakdown);
            plot_bar_chart(data.optimal_capacities);

            plot_sankey(data.sankey_data);
        });
    fetch('/get_plot_data/' + project_id + '/duration_curve')
        .then(response => response.json())
        .then(data => {
            plot_duration_curves(data.duration_curve);
        });
    fetch('/get_plot_data/' + project_id + '/emissions')
        .then(response => response.json())
        .then(data => {
            plot_co2_emissions(data.emissions);
        });
}

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
            removeLinksFromMap(map);
            put_links_on_map(links)
        })
        .catch((error) => {
            console.error("Error fetching data:", error);
        });
}


async function db_nodes_to_js(project_id, markers_only) {
    fetch("/db_nodes_to_js/" + project_id + '/' + markers_only)
        .then(response => response.json())
        .then(data => {
            if (data !== null) {
                map_elements = data.map_elements;
                is_load_center = data.is_load_center;
                load_legend();
                if (map_elements !== null) {
                    put_markers_on_map(map_elements, markers_only);
                }
            } else {
                map_elements = [];
            }
        });
}


async function consumer_to_db(project_id, href) {
    update_map_elements();
    const url = "/consumer_to_db/" + project_id;
    await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({map_elements: map_elements})
    }).then(() => {
        if (!href) {
            forward_if_consumer_selection_exists(project_id);
        } else {
            window.location.href = href;
        }
    });
}

function add_buildings_inside_boundary({boundariesCoordinates} = {}) {
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
            $("*").css('cursor', 'auto');
            const responseMsg = document.getElementById("responseMsg");
            responseMsg.innerHTML = res.msg;
            if (res.executed === false) {
            } else {
                responseMsg.innerHTML = "";
                Array.prototype.push.apply(map_elements, res.new_consumers);
                put_markers_on_map(res.new_consumers, true);
            }
            unique_map_elements();
        })
        .catch((error) => {
            console.error("Error fetching data:", error);
        });
}

async function remove_buildings_inside_boundary({boundariesCoordinates} = {}) {
    $("*").css("cursor", "wait");

    try {
        const response = await fetch("/remove_buildings_inside_boundary", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                boundary_coordinates: boundariesCoordinates,
                map_elements: map_elements,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const res = await response.json();

        map_elements = res.map_elements;
        remove_marker_from_map();
        put_markers_on_map(map_elements, true);
    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    } finally {
        $("*").css('cursor', 'auto');
    }
}


async function redirect(href) {
    window.location.href = href;
}


async function save_energy_system_design(href) {
    const url = "save_energy_system_design/";
    const data = {
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
                'min_load': dieselGensetMinLoad.value / 100,
                'max_efficiency': dieselGensetMaxEfficiency.value / 100,
                'max_load': dieselGensetMaxLoad.value / 100,
                'min_efficiency': dieselGensetMinEfficiency.value / 100,
            }
        },
        battery: {
            'settings': {
                'is_selected': selectBattery.checked,
                'design': batteryDesign.checked,
            },
            'parameters': {
                'nominal_capacity': batteryNominalCapacity.value,
                'lifetime': batteryLifetime.value,
                'capex': batteryCapex.value,
                'opex': batteryOpex.value,
                'soc_min': batterySocMin.value / 100,
                'soc_max': batterySocMax.value / 100,
                'c_rate_in': batteryCrateIn.value,
                'c_rate_out': batteryCrateOut.value,
                'efficiency': batteryEfficiency.value / 100,
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
                'efficiency': inverterEfficiency.value / 100,
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
                'efficiency': rectifierEfficiency.value / 100
            },
        },
        shortage: {
            'settings': {
                'is_selected': selectShortage.checked,
            },
            'parameters': {
                'max_shortage_total': shortageMaxTotal.value / 100,
                'max_shortage_timestep': shortageMaxTimestep.value / 100,
                'shortage_penalty_cost': shortagePenaltyCost.value
            },
        },
    };

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error("HTTP error " + response.status);
        }

        await response.json(); // Wait for response to be parsed
        if (href.length > 0) {
            window.location.href = href; // navigate after fetch request is complete
        }
    } catch (err) {
        console.log("An error occurred while saving the energy system design.");
    }
}

let hasRetried = false;

async function load_results(project_id) {
    try {
        const url = "load_results/" + project_id;
        const response = await fetch(url);

        if (!response.ok) {
            console.error("Network response was not ok");
            return;
        }

        const results = await response.json();

        if (results['n_consumers'] > 0) {
            document.getElementById('noResults').style.display = 'none';
            document.getElementById("nConsumers").innerText = Number(results['n_consumers']) - Number(results['n_shs_consumers']);
            document.getElementById("nGridConsumers").innerText = Number(results['n_consumers']) - Number(results['n_shs_consumers']);
            document.getElementById("nShsConsumers").innerText = results['n_shs_consumers'];
            document.getElementById("nPoles").innerText = results['n_poles'];
            document.getElementById("lengthDistributionCable").innerText = results['length_distribution_cable'];
            document.getElementById("averageLengthDistributionCable").innerText = results['average_length_distribution_cable'];
            document.getElementById("lengthConnectionCable").innerText = results['length_connection_cable'];
            document.getElementById("averageLengthConnectionCable").innerText = results['average_length_connection_cable'];
            document.getElementById("res").innerText = results['res'];
            document.getElementById("surplusRate").innerText = results['surplus_rate'];
            document.getElementById("shortageTotal").innerText = results['shortage_total'];
            document.getElementById("lcoe").innerHTML = results['lcoe'].toString() + " ¢<sub class='sub'>USD</sub>/kWh";
            document.getElementById("gridLcoe").innerText = results['gridLcoe'];
            document.getElementById("esLcoe").innerText = results['esLcoe'];
            document.getElementById("totalConsumption").innerText = results['total_annual_consumption'];
            document.getElementById("totalUpfrontInvestmentCost").innerText = results['upfront_invest_total'];
            document.getElementById("totalUpfrontInvestmentCost2").innerText = results['upfront_invest_total'];
            document.getElementById("consumptionPerConsumer").innerText = results['average_annual_demand_per_consumer'];
            document.getElementById("time").innerText = results['time'];
            document.getElementById("capPV").innerText = results['pv_capacity'];
            document.getElementById("capBattery").innerText = results['battery_capacity'];
            document.getElementById("capDiesel").innerText = results['diesel_genset_capacity'];
            document.getElementById("capPV2").innerText = results['pv_capacity'];
            document.getElementById("capBattery2").innerText = results['battery_capacity'];
            document.getElementById("capDiesel2").innerText = results['diesel_genset_capacity'];
            document.getElementById("capInverter").innerText = results['inverter_capacity'];
            document.getElementById("capRect").innerText = results['rectifier_capacity'];
            document.getElementById("res2").innerText = results['res'];
            document.getElementById("fuelConsumption").innerText = results['fuel_consumption'];
            document.getElementById("coTwoEmissions").innerText = results['co2_emissions'];
            document.getElementById("coTwoSavings").innerText = results['co2_savings'];
            document.getElementById("totalConsumption2").innerText = results['total_annual_consumption'];
            document.getElementById("peakDemand").innerText = results['peak_demand'];
            document.getElementById("baseLoad").innerText = results['base_load'];
            document.getElementById("consumptionPerConsumer2").innerText = results['average_annual_demand_per_consumer'];
            document.getElementById("shortage").innerText = results['shortage_total'];
            document.getElementById("surplus").innerText = results['surplus_rate'];
            document.getElementById("max_shortage").innerText = results['max_shortage'];
            document.getElementById('GridUpfrontInvestmentCost').innerText = results['upfront_invest_grid'];
            document.getElementById('DieselUpfrontInvestmentCost').innerText = results['upfront_invest_diesel_gen'];
            document.getElementById('PVUpfrontInvestmentCost').innerText = results['upfront_invest_pv'];
            document.getElementById('InverterUpfrontInvestmentCost').innerText = results['upfront_invest_inverter'];
            document.getElementById('RectifierUpfrontInvestmentCost').innerText = results['upfront_invest_rectifier'];
            document.getElementById('BatteryUpfrontInvestmentCost').innerText = results['upfront_invest_battery'];
            document.getElementById('fuelCost').innerText = results['cost_fuel'];
            document.getElementById('cost_grid').innerText = results['cost_grid'];
            document.getElementById('epc_pv').innerText = results['epc_pv'];
            document.getElementById('epc_inverter').innerText = results['epc_inverter'];
            document.getElementById('epc_rectifier').innerText = results['epc_rectifier'];
            document.getElementById('epc_diesel_genset').innerText = results['epc_diesel_genset'];
            document.getElementById('epc_battery').innerText = results['epc_battery'];
            document.getElementById('epc_total').innerText = results['epc_total'];
            document.getElementById('LCOE').innerHTML = results['lcoe'].toString() + " Cent<sub class='sub'>USD</sub>/kWh";
            db_nodes_to_js(project_id, false);
            db_links_to_js(project_id);

            if (results['lcoe'] === null || results['lcoe'] === undefined || results['lcoe'].includes('None')) {
                if (results['responseMsg'].length === 0) {
                    document.getElementById('responseMsg').innerHTML = 'Something went wrong. There are no results of the energy system optimization.';
                } else {
                    document.getElementById('responseMsg').innerHTML = results['responseMsg'];
                }
            } else {
                plot();
            }

        } else {
            document.getElementById('dashboard').style.display = 'none';
            document.getElementById('map').style.display = 'none';
            document.getElementById('noResults').style.display = 'block';

            const res = await fetch("has_pending_task/" + project_id, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await res.json();

            if (data.has_pending_task === true) {
                document.getElementById("pendingTaskMSG").innerText = 'Calculation is still running.';
            } else {
                document.getElementById("pendingTaskMSG").innerText = 'There is no ongoing calculation.\n' +
                    'Do you want to start a new calculation?';
            }
        }

    } catch (error) {
        console.error("An error occurred:", error);

        if (!hasRetried) {
            hasRetried = true;
            console.log("Retrying function load_results");
            await load_results(project_id);
        } else {
            console.log("Already retried once. Not retrying again.");
        }
    }
}


/************************************************************/
/*                    User Registration                     */

/************************************************************/


async function add_user_to_db() {
    $("*").css('cursor', 'wait');

    const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
    await sleep(3000); // Pause for 3 seconds (3000 milliseconds)

    if (userPassword2.value !== userPassword3.value) {
        document.getElementById("responseMsg2").innerHTML = 'The passwords do not match';
        document.getElementById("responseMsg2").style.color = 'red';
    } else {
        try {
            const response = await fetch("add_user_to_db/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    'email': userEmail2.value,
                    'password': userPassword2.value,
                    'captcha_input': captcha_input2.value,
                    'hashed_captcha': hashedCaptcha
                }),
            });
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            const responseData = await response.json();
            document.getElementById("responseMsg2").innerHTML = responseData.msg;
            const fontcolor = responseData.validation ? 'green' : 'red';
            document.getElementById("responseMsg2").style.color = fontcolor;
        } catch (error) {
            console.error("There was a problem with the fetch operation:", error.message);
        }
    }
    $("*").css('cursor', 'auto');
}


async function change_email() {
    if (userEmail1.value !== userEmail2.value) {
        document.getElementById("responseMsg1").innerHTML = 'The emails do not match';
        document.getElementById("responseMsg1").style.color = 'red';
    } else {
        $("*").css("cursor", "wait");

        try {
            const response = await fetch("change_email/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    email: userEmail1.value,
                    password: userPassword.value,
                    remember_me: false
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const responseData = await response.json();
            document.getElementById("responseMsg1").innerHTML = responseData.msg;
            const fontcolor = responseData.validation ? 'green' : 'red';
            document.getElementById("responseMsg1").style.color = fontcolor;

            if (responseData.validation === true) {
                await new Promise(r => setTimeout(r, 3000));
                logout();
            }

        } catch (error) {
            console.error("There was a problem with the fetch operation:", error.message);
        } finally {
            $("*").css("cursor", "auto");
        }
    }
}


async function change_pw() {
    if (newUserPassword1.value != newUserPassword2.value) {
        document.getElementById("responseMsg2").innerHTML = 'The passwords do not match';
        document.getElementById("responseMsg2").style.color = 'red';
    } else {
        $("*").css("cursor", "wait");

        try {
            const response = await fetch("change_pw/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    new_password: newUserPassword1.value,
                    old_password: oldUserPassword.value
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const responseData = await response.json();
            document.getElementById("responseMsg2").innerHTML = responseData.msg;
            const fontcolor = responseData.validation ? 'green' : 'red';
            document.getElementById("responseMsg2").style.color = fontcolor;

            if (responseData.validation === true) {
                await new Promise(r => setTimeout(r, 3000));
                logout();
            }

        } catch (error) {
            console.error("There was a problem with the fetch operation:", error.message);
        } finally {
            $("*").css("cursor", "auto");
        }
    }
}


async function delete_account() {
    $("*").css("cursor", "wait");

    try {
        const response = await fetch("delete_account/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                password: Password.value
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const responseData = await response.json();
        document.getElementById("responseMsg3").innerHTML = responseData.msg;
        const fontcolor = responseData.validation ? 'green' : 'red';
        document.getElementById("responseMsg3").style.color = fontcolor;

        if (responseData.validation === true) {
            await new Promise(r => setTimeout(r, 3000));
            logout();
        }

    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    } finally {
        $("*").css("cursor", "auto");
    }
}


async function login() {
    try {
        const response = await fetch("login/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                email: userEmail.value,
                password: userPassword.value,
                remember_me: isEnabled.value
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const responseData = await response.json();

        document.getElementById("userPassword").value = '';
        if (responseData.validation === true) {
            document.getElementById("userEmail").value = '';
            location.reload();
        } else {
            document.getElementById("responseMsg").innerHTML = responseData.msg;
            document.getElementById("responseMsg").style.color = 'red';
        }

    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    }
}


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


async function consent_cookie() {
    try {
        const response = await fetch("consent_cookie/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        document.getElementById('consentCookie').style.display = 'none';

    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    }
}


async function anonymous_login() {
    try {
        const response = await fetch("anonymous_login/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                'captcha_input': captcha_input3.value,
                'hashed_captcha': hashedCaptcha
            })
        });

        const data = await response.json();

        document.getElementById("responseMsg3").innerHTML = data.msg;
        if (data.validation === true) {
            window.location.href = window.location.origin;
        } else {
            document.getElementById("responseMsg3").style.color = 'red';
        }

    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    }
}


async function logout() {
    try {
        const response = await fetch("logout/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
        });

        if (response.ok) {
            window.location.href = window.location.origin;
        } else {
            console.error("Failed to log out");
        }

    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    }
}


async function save_project_setup(project_id, href) {
    event.preventDefault(); // prevent the link from navigating immediately
    const url = "save_project_setup/" + project_id;
    const data = {
        page_setup: {
            'project_name': projectName.value,
            'project_description': projectDescription.value.trim(),
            'interest_rate': interestRate.value,
            'project_lifetime': projectLifetime.value,
            'start_date': "2022-01-01",
            'temporal_resolution': 1,
            'n_days': nDays.value,
        }
    };
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data),
        });
        if (!response.ok) {
            throw new Error("HTTP error " + response.status);
        }
        window.location.href = href; // navigate after fetch request is complete
    } catch (err) {
        console.log("An error occurred while saving the project setup:", err);
    }
}

async function save_grid_design(href) {
    try {
        let shs_max_grid_cost_value;
        if (document.getElementById('shs_max_grid_cost').disabled) {
            shs_max_grid_cost_value = 999;
        } else {
            shs_max_grid_cost_value = document.getElementById('shs_max_grid_cost').value;
        }

        await fetch("save_grid_design/", {
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
                    'shs_max_grid_cost': shs_max_grid_cost_value,
                }
            })
        });

        window.location.href = href; // navigate after fetch request is complete
    } catch (err) {
        console.log('Fetch API error -', err);
    }
}


function save_demand_estimation(href) {
    let custom_calibration = document.getElementById("toggleswitch").checked;
    let use_custom_shares = document.getElementById("use_custom_shares").checked;
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
                'use_custom_shares': use_custom_shares,
                'custom_share_1': custom_share_1.value,
                'custom_share_2': custom_share_2.value,
                'custom_share_3': custom_share_3.value,
                'custom_share_4': custom_share_4.value,
                'custom_share_5': custom_share_5.value,
            }
        })
    }).then(r => window.location.href = href)
}


function load_previous_data(page_name) {
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
                if (results !== null && Object.keys(results).length > 1) {
                    document.getElementById("projectName").value = results['project_name'];
                    document.getElementById("projectDescription").value = results['project_description'];
                    document.getElementById("interestRate").value = results['interest_rate'];
                    document.getElementById("projectLifetime").value = results['project_lifetime'];
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
                    if (results['shs_max_grid_cost'] === 999 || isNaN(results['shs_max_grid_cost']) || results['shs_max_grid_cost'] === null) {
                        document.getElementById('shs_max_grid_cost').value = '';
                        document.getElementById('selectShsBox').classList.add('box--not-selected');
                        document.getElementById('selectShs').checked = false;
                        document.getElementById('lblShsLifetime').classList.add('disabled');
                        document.getElementById('shsLifetimeUnit').classList.add('disabled');
                        document.getElementById('shs_max_grid_cost').disabled = true;
                    } else {
                        document.getElementById('shs_max_grid_cost').value = results['shs_max_grid_cost'];
                        document.getElementById('selectShsBox').classList.remove('box--not-selected');
                        document.getElementById('selectShs').checked = true;
                        document.getElementById('lblShsLifetime').classList.remove('disabled');
                        document.getElementById('shsLifetimeUnit').classList.remove('disabled');
                        document.getElementById('shs_max_grid_cost').disabled = false;
                    }
                }
                change_shs_box_visibility();
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
                    document.getElementById("use_custom_shares").checked = results['use_custom_shares'];

                    let accordionItem2 = new bootstrap.Collapse(document.getElementById('collapseTwo'),
                        {toggle: false});
                    if (results['custom_calibration'] == true) {
                        accordionItem2.show();
                        const radioButton2 = document.querySelector(`input[name="options2"][id="option${results['calibration_options'] + 6}"]`);
                        if (radioButton2) {
                            radioButton2.checked = true;
                            if (results['calibration_options'] === 2) {
                                document.getElementById("maximum_peak_load").disabled = false;
                                document.getElementById("average_daily_energy").disabled = true;
                            }
                        }
                    } else {
                        accordionItem2.hide();
                    }

                    if (results['use_custom_shares'] == true) {
                        document.getElementById("custom_share_1").value = results['custom_share_1'];
                        document.getElementById("custom_share_2").value = results['custom_share_2'];
                        document.getElementById("custom_share_3").value = results['custom_share_3'];
                        document.getElementById("custom_share_4").value = results['custom_share_4'];
                        document.getElementById("custom_share_5").value = results['custom_share_5'];
                    } else {

                    }

                    const radioButton = document.querySelector(`input[name="options"][id="option${results['household_option'] + 1}"]`);
                    if (radioButton) {
                        radioButton.checked = true;
                    }
                }
            }
        }
    } else if (page_name.includes("energy_system_design")) {
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
            refreshBlocksOnDiagramOnLoad();
            check_box_visibility('shortage');
        }
    }
}


function show_email_and_project_in_navbar(project_id = null) {
    fetch("query_account_data/", {
        method: "POST",
        headers: {"Content-Type": "application/json",},
        body: JSON.stringify({
            'project_id': project_id
        }),
    })
        .then(response => response.json())
        .then(data => {
            const showMailElement = document.getElementById("showMail");
            const showProjectElement = document.getElementById("showProject");
            if (showMailElement) {
                showMailElement.innerHTML = data.email;
            }
            if (showProjectElement && data.project_name) {
                showProjectElement.innerHTML = "     Project: " + data.project_name;
            }
        });
}


async function redirect_if_cookie_is_missing(access_token, consent_cookie) {
    let has_access_token = (access_token === true || access_token === 'true');
    let has_consent_cookie = (consent_cookie === true || consent_cookie === 'true');

    try {
        const response = await fetch("has_cookie/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                'access_token': has_access_token,
                'consent_cookie': has_consent_cookie
            }),
        });

        const responseData = await response.json();

        if (!responseData) {
            logout();
            window.location.href = window.location.origin;
        }

        await renewToken();

    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    }
}


async function remove_project(project_id) {
    try {
        const response = await fetch("remove_project/" + project_id, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });

        if (response.ok) {
            window.location.href = window.location.origin;
        } else {
            console.error("Failed to remove the project. Status:", response.status);
        }
    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    }
}


let shouldStop = false;


async function wait_for_results(project_id, task_id, time, model, opt_iter) {
    // Get the current URL
    var url = window.location.href;

    // If the url includes /calculating, proceed with the request
    if (url.includes("/calculating") && !shouldStop) {
        try {
            const response = await fetch("waiting_for_results/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({'project_id': project_id, 'task_id': task_id, 'time': time, 'model': model})
            });

            if (response.ok) {
                const res = await response.json();

                if (res.finished === true) {
                    opt_iter = opt_iter + 1;
                    if (opt_iter <= 0) {
                        start_calculation(project_id, opt_iter)
                    } else {
                        window.location.href = window.location.origin + '/simulation_results?project_id=' + project_id;
                    }
                } else if (!shouldStop) {
                    document.querySelector("#statusMsg").innerHTML = res.status;
                    renewToken();
                    wait_for_results(project_id, task_id, res.time, res.model, opt_iter);
                }
            } else {
                if (response.status === 303 || response.status === 422) {
                    shouldStop = true;
                    window.location.href = "/?internal_error";
                }
            }
        } catch (error) {
            console.error("There was a problem with the fetch operation:", error.message);
        }
    }
}


async function forward_if_no_task_is_pending(project_id) {
    try {
        const response = await fetch("forward_if_no_task_is_pending/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
        });

        if (response.ok) {
            const res = await response.json();

            if (res.forward === true) {
                window.location.href = window.location.origin + '/calculating?project_id=' + project_id;
            } else {
                document.getElementById('pendingTask').style.display = 'block';
            }
        } else {
            console.error("Server responded with a status:", response.status);
        }
    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    }
}


async function revoke_users_task() {
    try {
        const response = await fetch("revoke_users_task/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });

        if (response.ok) {
            document.getElementById('pendingTask').style.display = 'none';
        } else {
            console.error("Server responded with a status:", response.status);
        }
    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    }
}


function start_calculation(project_id, opt_iter = 0) {
    fetch("start_calculation/" + project_id, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        }
    })
        .then(response => response.json())
        .then(res => {
            if (res.redirect && res.redirect.length > 0) {
                document.getElementById('responseMsg').innerHTML =
                    'Input data is missing for the opt_models. It appears that you have not gone through all the pages to ' +
                    'enter the input data. You will be redirected to the corresponding page.';
                const baseURL = window.location.origin;
                document.getElementById('redirectLink').href = baseURL + res.redirect;
                document.getElementById('msgBox').style.display = 'block';
            } else {
                wait_for_results(project_id, res.task_id, 0, 'grid', opt_iter);
            }
        })
        .catch(error => console.error('There was an error!', error));
}


async function forward_if_consumer_selection_exists(project_id) {
    try {
        const response = await fetch("forward_if_consumer_selection_exists/" + project_id, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });

        if (response.ok) {
            const res = await response.json();
            if (res.forward === true) {
                window.location.href = window.location.origin + '/demand_estimation?project_id=' + project_id;
            } else {
                document.getElementById('responseMsg').innerHTML = 'No consumers are selected. You must select the geolocation of the consumers before you go to the next page.';
            }
        } else {
            console.error("Server responded with a status:", response.status);
        }
    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    }
}


async function send_email_notification(project_id, is_active) {
    try {
        const response = await fetch("/set_email_notification/" + project_id + '/' + is_active, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });

        if (!response.ok) {
            console.error("Server responded with a status:", response.status);
        }
    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    }
}


async function show_cookie_consent() {
    try {
        const response = await fetch("has_cookie/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({'access_token': false, 'consent_cookie': true})
        });

        const data = await response.json();

        if (data == false) {
            document.getElementById('consentCookie').style.display = 'block';
        } else {
            document.getElementById('consentCookie').style.display = 'none';
        }

    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    }
}


async function send_reset_password_email() {
    try {
        const response = await fetch("send_reset_password_email/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                'email': userEmail4.value,
                'captcha_input': captcha_input.value,
                'hashed_captcha': hashedCaptcha
            })
        });

        const data = await response.json();

        document.getElementById("responseMsg4").innerHTML = data.msg;
        let fontcolor;
        if (data.validation === true) {
            fontcolor = 'green';
        } else {
            fontcolor = 'red';
        }
        document.getElementById("responseMsg4").style.color = fontcolor;

        if (data.validation === true) {
            await new Promise(r => setTimeout(r, 3000));
            document.getElementById('forgotPassword').style.display = 'none';
        }

    } catch (error) {
        console.error("There was a problem with the fetch operation:", error.message);
    }
}


function reset_pw(guid) {
    if (newUserPassword1.value !== newUserPassword2.value) {
        document.getElementById("responseMsg2").innerHTML = 'The passwords do not match';
        document.getElementById("responseMsg2").style.color = 'red';
        return;
    }

    fetch("reset_password", {
        method: "POST",
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            guid: guid,
            password: newUserPassword1.value
        })
    })
        .then(response => response.json())
        .then(async (data) => {
            document.getElementById("responseMsg2").innerHTML = data.msg;
            let fontcolor = data.validation ? 'green' : 'red';
            document.getElementById("responseMsg2").style.color = fontcolor;

            if (data.validation) {
                await new Promise(r => setTimeout(r, 3000));
                window.location.href = window.location.origin;
            }
        })
        .catch(error => {
            console.error("There was an error:", error);
        });
}


function create_example_project() {
    fetch("/example_model/")
        .then(res => {
            if (res.ok) {  // Check if the fetch was successful
                window.location.reload();  // Reload the page
            } else {
                console.error('Failed to fetch example model. Status:', res.status);
            }
        })
        .catch(err => {
            console.error('Error fetching example model:', err);
        });
}


function show_video_tutorial() {
    fetch("/show_video_tutorial/")
        .then(response => response.json())
        .then(res => {
            let show_tutorial = res;
            if (show_tutorial === true) {
                document.getElementById('videoTutorial').style.display = 'block';
            } else {
                document.getElementById('videoTutorial').style.display = 'none';
            }
        })
}

function deactivate_video_tutorial() {
    fetch("/deactivate_video_tutorial/")
}

function copyProject(url) {
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                location.reload();
            } else {
                alert('Failed to copy project');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred');
        });
}


document.addEventListener('DOMContentLoaded', function () {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('.icon[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl, {
            trigger: 'hover click'
        });
    });
});


const img = document.getElementById("captcha_img");
const img2 = document.getElementById("captcha_img2");
const img3 = document.getElementById("captcha_img3");
let hashedCaptcha;

function get_captcha() {
    fetch("/get_captcha")
        .then(response => response.json())
        .then(data => {
            img.src = "data:image/jpeg;base64," + data.img;
            img2.src = "data:image/jpeg;base64," + data.img;
            img3.src = "data:image/jpeg;base64," + data.img;
            hashedCaptcha = data.hashed_captcha;
        });
}


async function sendMail() {
    // Getting values from the HTML elements
    const from_address = document.getElementById("from_address").value;
    const subject = document.getElementById("subject").value;
    const body = document.getElementById("body").value;

    function handleError() {
        const responseMsgElement = document.getElementById("responseMsg");
        responseMsgElement.innerText = "Something went wrong";
        responseMsgElement.style.color = "red";
    }

    // Creating the mail object
    const mail = {
        from_address: from_address,
        subject: subject,
        body: body
    };

    // Reference to the responseMsg element
    const responseMsgElement = document.getElementById("responseMsg");

    // Sending the mail object to the FastAPI route
    try {
        const response = await fetch("/send_mail_route/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(mail)
        });

        if (response.status === 200) {
            const result = await response.json();
            if (result.message === "Success") {
                responseMsgElement.innerText = "Email successfully sent";
                responseMsgElement.style.color = "green";

                // Redirect to the base URL after 1.5 seconds
                setTimeout(() => {
                    window.location.href = "/";
                }, 2500);
            } else {
                handleError();
            }
        } else {
            handleError();
        }
    } catch (error) {
        handleError();
    }
}