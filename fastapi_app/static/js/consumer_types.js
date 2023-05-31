let consumer_list = {
    'H': 'Houshold',
    'E': 'Enterprise',
    'P': 'Public Service'};
let consumer_type = "H";
(function () {
    let option_consumer =  '';
    for(let consumer_code in consumer_list){
        let selected = (consumer_code == consumer_type) ? ' selected' : '';
        option_consumer += '<option value="'+consumer_code+'"'+selected+'>'+consumer_list[consumer_code]+'</option>';}
    document.getElementById('consumer').innerHTML = option_consumer;
})();

let public_service_list = {
'group1' : 'Public Health Centre',
'group2' : 'Public Clinic',
'group3' : 'Public CHPS',
'group4' : 'School',
'group5' : 'Cell Tower',
'group6' : 'Street Light'
}

let enterprise_list = {

 'group1' :'Groceries',
 'group2' :'Restaurant',
 'group3' :'Bar',
 'group4' :'Drinks',
 'group5' :'Fruits or vegetables',
 'group6' :'Tailoring',
 'group7' :'Beauty or Hair',
 'group8' :'Metalworks',
 'group9' :'Car or Motorbike Repair',
 'group10' :'Carpentry',
 'group11' :'Laundry',
 'group12' :'Cycle Repair',
 'group13' :'Shoemaking',
 'group14' :'Medical',
 'group15' :'Clothes and accessories',
 'group16' :'Electronics',
 'group17' :'Retail or Other',
 'group18' :'Agricultural',
 'group19' :'Mobile or Electronics Repair',
 'group20' :'Digital Other',
 'group21' :'Cybercafé',
 'group22' :'Cinema or Betting',
 'group23' :'Photostudio',
 'group24' :'Mill or Thresher or Grater',
 'group25' :'Agricultural or Other'};

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
    'group11': 'Drill (0.4kW)',
    'group12': 'Welder (5.25kW)',
    'group13': 'Angle Grinder (2kW)'
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

let markerConsumerSelected = new L.Icon({
  iconUrl: "fastapi_app/static/assets/icons/i_consumer_selected.svg",
  iconSize: [12, 12],
});


let marker
let old_marker

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
        L.marker([markerLatLng.lat, markerLatLng.lng], {icon: markerConsumerSelected,})
        .on('click', markerOnClick).addTo(map);
        document.getElementById('longitude').value = marker.longitude;
        document.getElementById('latitude').value = marker.latitude;
        if (marker.consumer_type === 'household') {
           document.getElementById('consumer').value = 'H';
           document.getElementById('enterprise').disabled = true;
           document.getElementById('enterprise').value = '';
           deactivate_large_loads();

        }
        else if (marker.consumer_type === 'enterprise'){
            dropDownMenu(enterprise_list);
            document.getElementById('consumer').value = 'E';
            document.getElementById('enterprise').value = 'group1';
            activate_large_loads();
        }
        else if (marker.consumer_type === 'public_service'){
            dropDownMenu(public_service_list);
            deactivate_large_loads()

        }
        document.getElementById('consumer').disabled = false;
        document.getElementById('longitude').disabled = false;
        document.getElementById('latitude').disabled = false;
    }
  }
});
}

function update_map_elements(){
        marker.longitude = parseFloat(document.getElementById('longitude').value);
        marker.latitude = parseFloat(document.getElementById('latitude').value);
        if (document.getElementById('consumer').value === 'H') {
           marker.consumer_type = 'household';
           marker.consumer_detail = 'default';
        }
        else {
            marker.consumer_type = 'enterprise';
            let key = document.getElementById('enterprise').value;
            marker.consumer_detail = enterprise_list[key];
        }
    map_elements.push(marker);
    map.eachLayer(function (layer) {
        if (layer instanceof L.Marker) {
        let markerLatLng = layer.getLatLng();
        if (markerLatLng.lat === old_marker.latitude && markerLatLng.lng === old_marker.longitude) {
            map.removeLayer(layer);
            L.marker([marker.latitude, marker.longitude], {icon: markerConsumer,})
            .on('click', markerOnClick).addTo(map);
        }
    }})}

function move_marker(){
    old_marker = JSON.parse(JSON.stringify(marker));
    marker.longitude = parseFloat(document.getElementById('longitude').value);
    marker.latitude = parseFloat(document.getElementById('latitude').value);
    map.eachLayer(function (layer) {
        if (layer instanceof L.Marker) {
        let markerLatLng = layer.getLatLng();
        if (markerLatLng.lat === old_marker.latitude && markerLatLng.lng === old_marker.longitude) {
            map.removeLayer(layer);
            L.marker([marker.latitude, marker.longitude], {icon: markerConsumerSelected,})
            .on('click', markerOnClick).addTo(map);
        }
    }})
}

document.getElementById('latitude').addEventListener('change', move_marker);
document.getElementById('longitude').addEventListener('change', move_marker);

function deleteAllElements() {
    var listDiv = document.getElementById('load_list');
    listDiv.innerHTML = '';
}


function activate_large_loads() {
    deleteAllElements();
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