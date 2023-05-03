let consumer_list = {'H': 'Houshold', 'E': 'Enterprise'};
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



document.getElementById('consumer').addEventListener('change', function() {
    if (this.value === 'H') {
        document.getElementById('enterprise').value = '';
        document.getElementById('enterprise').disabled = true;
    } else {
        document.getElementById('enterprise').innerHTML = enterpise_option;
        document.getElementById('enterprise').value = 'group1';
        document.getElementById('enterprise').disabled = false;
    }
});
document.getElementById('enterprise').disabled = true;
document.getElementById('consumer').disabled = true;
document.getElementById('enterprise').value = '';
document.getElementById('consumer').value = '';

var markerConsumerSelected = new L.Icon({
  iconUrl: "fastapi_app/static/assets/icons/i_consumer_selected.svg",
  iconSize: [12, 12],
});

function get_consumer_parameters() {}

let marker
let old_marker

function markerOnClick(e){
    L.DomEvent.stopPropagation(e);
    update_map_elements();
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
        document.getElementById('floor_area').value = marker.surface_area;
        if (marker.consumer_type == 'household') {
           document.getElementById('consumer').value = 'H';
           document.getElementById('enterprise').disabled = true;
           document.getElementById('enterprise').value = '';
        }
        else {
            document.getElementById('consumer').value = 'E';
            document.getElementById('enterprise').disabled = false;
            document.getElementById('enterprise').value = 'group1';
        }
        document.getElementById('consumer').disabled = false;
        document.getElementById('longitude').disabled = false;
        document.getElementById('latitude').disabled = false;
        document.getElementById('floor_area').disabled = false;
    }
  }
});
}

function update_map_elements(){
    if (marker)
    {
        marker.longitude = parseFloat(document.getElementById('longitude').value);
        marker.latitude = parseFloat(document.getElementById('latitude').value);
        marker.surface_area = parseFloat(document.getElementById('floor_area').value);
        if (document.getElementById('consumer').value === 'H') {
           marker.consumer_type = 'household';
        }
        else {
                marker.consumer_type = 'enterprise';
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
    }})}}

