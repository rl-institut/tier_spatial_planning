let consumer_list = {'H': 'Houshold', 'E': 'Enterprise', 'P': 'Public Service', 'O': 'Ohter'};
let consumer_type = "H";
(function () {
    let option_consumer =  '';
    for(let consumer_code in consumer_list){
        let selected = (consumer_code == consumer_type) ? ' selected' : '';
        option_consumer += '<option value="'+consumer_code+'"'+selected+'>'+consumer_list[consumer_code]+'</option>';}
    document.getElementById('consumer').innerHTML = option_consumer;
})();

let enterprise_list = {'group1': 'Bakery', 'group2': 'Gas Station'};
let enterpise_option =  '';
for(let enterprise_code in enterprise_list){
    let selected = (enterprise_code == consumer_type) ? ' selected' : '';
    enterpise_option += '<option value="'+enterprise_code+'"'+selected+'>'+enterprise_list[enterprise_code]+'</option>';}

let large_load_list = {'W': 'Welder', 'M': 'Motor', 'I': 'Milling'};
let large_load_type = "M";
(function () {
    let option_load =  '';
    for(let load_code in large_load_list){
        let selected = (load_code == large_load_type) ? ' selected' : '';
        option_load += '<option value="'+load_code+'"'+selected+'>'+large_load_list[load_code]+'</option>';}
    document.getElementById('loads').innerHTML = option_load;
})();

document.getElementById('loads').disabled = true;
document.getElementById('loads').value = "";
document.getElementById('number_loads').disabled = true;

document.getElementById('consumer').addEventListener('change', function() {
    if (this.value === 'H') {
        document.getElementById('enterprise').value = '';
        document.getElementById('enterprise').disabled = true;
        document.getElementById('loads').disabled = true;
        document.getElementById('loads').value = '';
        document.getElementById('add').disabled = true;
        document.getElementById('number_loads').disabled = true;
        deleteAllElements();
    } else {
        document.getElementById('enterprise').innerHTML = enterpise_option;
        document.getElementById('enterprise').value = 'group1';
        document.getElementById('enterprise').disabled = false;
        document.getElementById('loads').disabled = false;
        document.getElementById('loads').value = 'W';
        document.getElementById('add').disabled = false;
        document.getElementById('number_loads').disabled = false;
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
           document.getElementById('loads').disabled = true;
           document.getElementById('loads').value = "";
           document.getElementById('add').disabled = true;
           document.getElementById('number_loads').disabled = true;

        }
        else {
            document.getElementById('consumer').value = 'E';
            document.getElementById('enterprise').disabled = false;
            document.getElementById('enterprise').value = 'group1';
           document.getElementById('loads').disabled = false;
           document.getElementById('add').disabled = false;
           document.getElementById('number_loads').disabled = false;
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
