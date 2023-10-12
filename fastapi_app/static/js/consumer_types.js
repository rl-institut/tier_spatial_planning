let consumer_list = {
    'H': 'Household',
    'E': 'Enterprise',
    'P': 'Public Service',
};
let consumer_type = "H";

(function () {
    let option_consumer = '';
    for (let consumer_code in consumer_list) {
        let selected = (consumer_code == consumer_type) ? ' selected' : '';
        option_consumer += '<option value="' + consumer_code + '"' + selected + '>' + consumer_list[consumer_code] + '</option>';
    }
    document.getElementById('consumer').innerHTML = option_consumer;
})();


let public_service_list = {
'group1' : 'Health_Health Centre',
'group2' : 'Health_Clinic',
'group3' : 'Health_CHPS',
'group4' : 'Education_School',
'group5' : 'Education_School_noICT'
}

let enterprise_list = {

 'group1' :'Food_Groceries',
 'group2' :'Food_Restaurant',
 'group3' :'Food_Bar',
 'group4' :'Food_Drinks',
 'group5' :'Food_Fruits or vegetables',
 'group6' :'Trades_Tailoring',
 'group7' :'Trades_Beauty or Hair',
 'group8' :'Trades_Metalworks',
 'group9' :'Trades_Car or Motorbike Repair',
 'group10' :'Trades_Carpentry',
 'group11' :'Trades_Laundry',
 'group12' :'Trades_Cycle Repair',
 'group13' :'Trades_Shoemaking',
 'group14' :'Retail_Medical',
 'group15' :'Retail_Clothes and accessories',
 'group16' :'Retail_Electronics',
 'group17' :'Retail_Other',
 'group18' :'Retail_Agricultural',
 'group19' :'Digital_Mobile or Electronics Repair',
 'group20' :'Digital_Digital Other',
 'group21' :'Digital_Cybercafé',
 'group22' :'Digital_Cinema or Betting',
 'group23' :'Digital_Photostudio',
 'group24' :'Agricultural_Mill or Thresher or Grater',
 'group25' :'Agricultural_Other'};

let enterpise_option =  '';

function dropDownMenu(dropdown_list) {
    enterpise_option =  '';
    for(let enterprise_code in dropdown_list){
    let selected = (enterprise_code == consumer_type) ? ' selected' : '';
    enterpise_option += '<option value="'+enterprise_code+'"'+selected+'>'+dropdown_list[enterprise_code]+'</option>';
    document.getElementById('enterprise').innerHTML = enterpise_option;
    document.getElementById('enterprise').disabled = false;}
}

let large_load_list = {
    'group1': 'Milling Machine (7.5kW)',
    'group2': 'Crop Dryer (8kW)',
    'group3': 'Thresher (8kW)',
    'group4': 'Grinder (5.2kW)',
    'group5': 'Sawmill (2.25kW)',
    'group6': 'Circular Wood Saw (1.5kW)',
    'group7': 'Jigsaw (0.4kW)',
    'group8': 'Drill (0.4kW)',
    'group9': 'Welder (5.25kW)',
    'group10': 'Angle Grinder (2kW)',
};
let large_load_type = "group1";

    let option_load =  '';
    for(let load_code in large_load_list){
        let selected = (load_code == large_load_type) ? ' selected' : '';
        option_load += '<option value="'+load_code+'"'+selected+'>'+large_load_list[load_code]+'</option>';}
    document.getElementById('loads').innerHTML = option_load;


document.getElementById('loads').disabled = true;
document.getElementById('loads').value = "";
document.getElementById('number_loads').disabled = true;

document.getElementById('consumer').addEventListener('change', function() {
    if (this.value === 'H') {
        document.getElementById('enterprise').value = '';
        document.getElementById('enterprise').disabled = true;
        deactivate_large_loads();
    }
    else if (this.value === 'E') {
        dropDownMenu(enterprise_list);
        document.getElementById('enterprise').innerHTML = enterpise_option;
        document.getElementById('enterprise').value = 'group1';
        document.getElementById('enterprise').disabled = false;
        activate_large_loads();
    }
    else if (this.value === 'P') {
        dropDownMenu(public_service_list);
        deactivate_large_loads();
    }
});
document.getElementById('enterprise').disabled = true;
document.getElementById('consumer').disabled = true;
document.getElementById('enterprise').value = '';
document.getElementById('consumer').value = '';
document.getElementById('shs_options').disabled = true;
document.getElementById('shs_options').value = '';


let markerConsumerSelected = new L.Icon({
  iconUrl: "fastapi_app/static/assets/icons/i_consumer_selected.svg",
  iconSize: [12, 12],
});

let markerPowerHouseSelected = new L.Icon({
  iconUrl: "fastapi_app/static/assets/icons/i_power_house_selected.svg",
  iconSize: [12, 12],
});


let marker
let old_marker

function getKeyByValue(object, value) {
  return Object.keys(object).find(key => object[key] === value);
}



function markerOnClick(e){
    L.DomEvent.stopPropagation(e);
    if (marker) {
    update_map_elements();
    }
    expandAccordionItem2();
    const index = map_elements.findIndex(obj => obj.latitude === e.latlng.lat && obj.longitude === e.latlng.lng);
    if (index >= 0) {
        marker = map_elements.splice(index, 1)[0];
        old_marker = JSON.parse(JSON.stringify(marker));
    }
    map.eachLayer(function (layer) {
    if (layer instanceof L.Marker) {
    let markerLatLng = layer.getLatLng();
    if (markerLatLng.lat === e.latlng.lat && markerLatLng.lng === e.latlng.lng) {
        map.removeLayer(layer);
        let markerIcon;
            if (marker.node_type === 'power-house') {
                    markerIcon = markerPowerHouseSelected;
            } else {
                markerIcon = markerConsumerSelected;
            }
        L.marker([markerLatLng.lat, markerLatLng.lng], {icon: markerIcon,})
        .on('click', markerOnClick).addTo(map);
        document.getElementById('longitude').value = marker.longitude;
        document.getElementById('latitude').value = marker.latitude;
        if (marker.node_type === 'power-house') {
           document.getElementById('consumer').value = '';
           document.getElementById('consumer').disabled = true;
           document.getElementById('shs_options').value = '';
           document.getElementById('shs_options').disabled = true;
           document.getElementById('enterprise').disabled = true;
           document.getElementById('enterprise').value = '';
           deactivate_large_loads();
        }
        else if (marker.consumer_type === 'household') {
           document.getElementById('consumer').value = 'H';
           document.getElementById('enterprise').disabled = true;
           document.getElementById('enterprise').value = '';
           document.getElementById('shs_options').disabled = false;
        document.getElementById('consumer').disabled = false;
           deactivate_large_loads();

        }
        else if (marker.consumer_type === 'enterprise'){
            dropDownMenu(enterprise_list);
            document.getElementById('consumer').value = 'E';
            let key = getKeyByValue(enterprise_list, marker.consumer_detail);
            document.getElementById('enterprise').value = key;
            document.getElementById('shs_options').disabled = false;
            document.getElementById('consumer').disabled = false;
            activate_large_loads();
            if (marker.custom_specification.length > 5) {
                activate_large_loads(false);
                fillList(marker.custom_specification);
                document.getElementById('toggleswitch2').checked = true;
                  const accordionItem3 = new bootstrap.Collapse(document.getElementById('collapseThree'), {
                    toggle: false
                  });
                accordionItem3.show();
            }
        }
        else if (marker.consumer_type === 'public_service'){
            dropDownMenu(public_service_list);
            document.getElementById('shs_options').disabled = false;
            document.getElementById('consumer').value = 'P';
            document.getElementById('consumer').disabled = false;
            let public_service_key2 = getKeyByValue(public_service_list, marker.consumer_detail);
            document.getElementById('enterprise').value = public_service_key2;
            document.getElementById('consumer').disabled = false;
            deactivate_large_loads()
        }
        if (marker.node_type !== 'power-house') {
            if (marker.shs_options == 0) {document.getElementById('shs_options').value = 'optimize';}
            else if (marker.shs_options == 1) {document.getElementById('shs_options').value = 'grid';}
            else if (marker.shs_options == 2) {document.getElementById('shs_options').value ='shs';}
        }
        document.getElementById('longitude').disabled = false;
        document.getElementById('latitude').disabled = false;

    }
  }
});
}

function update_map_elements(){
    let longitude = document.getElementById('longitude').value;
    let latitude = document.getElementById('latitude').value;
    let shs_options = document.getElementById('shs_options').value;
    let shs_value;
    let large_load_string = large_loads_to_string();

    switch (shs_options) {
        case 'optimize':
            shs_value = 0;
            break;
        case 'grid':
            shs_value = 1;
            break;
        case 'shs':
            shs_value = 2;
            break;
        default:
            shs_value = 0;
    }


    let selected_icon;

    if (longitude.length > 0 && latitude.length > 0) {
        marker.longitude = parseFloat(longitude);
        marker.latitude = parseFloat(latitude);
        marker.shs_options = parseInt(shs_value);
        marker.custom_specification = large_load_string;


        let consumerValue = document.getElementById('consumer').value;

        switch (consumerValue) {
            case 'H':
                marker.consumer_type = 'household';
                marker.consumer_detail = 'default';
                selected_icon = markerConsumer;
                break;
            case 'P':
                marker.consumer_type = 'public_service';
                let public_service_key = document.getElementById('enterprise').value;
                marker.consumer_detail = public_service_list[public_service_key];
                selected_icon = markerPublicservice;
                break;
            case 'E':
                marker.consumer_type = 'enterprise';
                let key = document.getElementById('enterprise').value;
                marker.consumer_detail = enterprise_list[key];
                selected_icon = markerEnterprise;
                break;
            case '':
                marker.node_type = 'power-house';
                marker.consumer_type = '';
                marker.consumer_detail = '';
                selected_icon = markerPowerHouse;
                break;
            default:
                console.error("Invalid consumer value: " + consumerValue);
        }

        if (marker.shs_options == 2) {selected_icon = markerShs;}
        map_elements.push(marker);

        map.eachLayer(function (layer) {
            if (layer instanceof L.Marker) {
                let markerLatLng = layer.getLatLng();
                if (markerLatLng.lat === old_marker.latitude && markerLatLng.lng === old_marker.longitude) {
                    map.removeLayer(layer);
                    L.marker([marker.latitude, marker.longitude], {icon: selected_icon})
                      .on('click', markerOnClick).addTo(map);
                }
            }
        });
    }
}

function move_marker() {
    old_marker = JSON.parse(JSON.stringify(marker));
    marker.longitude = parseFloat(document.getElementById('longitude').value);
    marker.latitude = parseFloat(document.getElementById('latitude').value);
    map.eachLayer(function (layer) {
        if (layer instanceof L.Marker) {
            let markerLatLng = layer.getLatLng();
            if (markerLatLng.lat === old_marker.latitude && markerLatLng.lng === old_marker.longitude) {
                map.removeLayer(layer);
                let markerIcon;
                if (marker.node_type === 'power-house') {
                    markerIcon = markerPowerHouseSelected;
                } else {
                    markerIcon = markerConsumerSelected;
                }
                L.marker([marker.latitude, marker.longitude], { icon: markerIcon })
                    .on('click', markerOnClick)
                    .addTo(map);
            }
        }
    });
}


document.getElementById('latitude').addEventListener('change', move_marker);
document.getElementById('longitude').addEventListener('change', move_marker);

function deleteAllElements() {
    var listDiv = document.getElementById('load_list');
    listDiv.innerHTML = '';
}


function activate_large_loads(delete_list_elements = true) {
    if (delete_list_elements == true) {deleteAllElements();}
    document.getElementById('loads').innerHTML = option_load;
    document.getElementById('loads').disabled = false;
    document.getElementById('add').disabled = false;
    document.getElementById('number_loads').disabled = false;
}


function deactivate_large_loads() {
    deleteAllElements();
    document.getElementById('loads').disabled = true;
    document.getElementById('loads').value = "";
    document.getElementById('add').disabled = true;
    document.getElementById('number_loads').disabled = true;
}


function large_loads_to_string() {
    let load_list = document.getElementById("load_list");
    let list_items = load_list.getElementsByTagName("div");
    let texts = [];
    for(let i = 0; i < list_items.length; i++) {
        let text = list_items[i].textContent.trim();
        text = text.replace('Delete', '').trim();
        texts.push(text);
    }
    let concatenated_text = texts.join(";");
    return concatenated_text;

}


function fillList(concatenated_text) {
    let texts = concatenated_text.split(";");
    for(let i = 0; i < texts.length; i++) {
        addElementToLargeLoadList(texts[i]);
    }
}


function addElementToLargeLoadList(customText) {
    var dropdown = document.getElementById('loads');
    var selectedValue = dropdown.options[dropdown.selectedIndex].text;
    var inputValue = document.getElementById('number_loads').value;
    var list = document.getElementById('load_list');
    var newItem = document.createElement('div');
    var newButton = document.createElement('button');
    newButton.classList.add('right-align');
    newButton.textContent = 'Delete';
    newButton.onclick = function() {
        list.removeChild(newItem);
    };
    if (customText) {
        newItem.textContent = customText + '    ';
    } else {
        newItem.textContent = inputValue + ' x ' + selectedValue + '    ';
    }
    newItem.appendChild(newButton);
    newItem.style.marginBottom = '10px';
    list.appendChild(newItem);
    if (!customText) {
        document.getElementById('number_loads').value = '1';
    }
};