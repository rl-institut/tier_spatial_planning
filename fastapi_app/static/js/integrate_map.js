// Create the map and set the center and zoom level

const nigeriaBounds = [
  [4.2, 2.7], // Southwest corner
  [13.9, 14.7] // Northeast corner
];

const map = L.map('map', {
  center: [9.8838, 5.9231],
  zoom: 6,
  maxBounds: nigeriaBounds,
  maxBoundsViscosity: 1.0
});


// Add the OSM map
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);
const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);
let polygonCoordinates = [];
let map_elements =[];

var markerConsumer = new L.Icon({
  iconUrl: "fastapi_app/static/assets/icons/i_consumer.svg",
  iconSize: [6, 6],
});

var markerPowerHouse = new L.Icon({
  iconUrl: "fastapi_app/static/assets/icons/i_power_house.svg",
  iconSize: [12, 12],
});

var markerPole = new L.Icon({
  iconUrl: "fastapi_app/static/assets/icons/i_pole.svg",
  iconSize: [10, 10],
});

var markerShs = new L.Icon({
  iconUrl: "fastapi_app/static/images/markers/markerShs.png",
  iconSize: [16, 16],
});


function drawMarker(latitude, longitude, type) {
  if (type === "consumer") {
    icon_type = markerConsumer;
  } else if (type === "pole") {
    icon_type = markerPole;
  } else if (type === "shs") {
    icon_type = markerShs;
  }
    L.marker([latitude, longitude], { icon: icon_type }).on('click', markerOnClick).addTo(map)
}

var icons = {
  'consumer': markerConsumer,
  'power-house': markerPowerHouse,
  'pole': markerPole,
  'shs': markerShs,
};


function zoomAll(map) {
  let latLonList = map_elements.map(obj => L.latLng(obj.latitude, obj.longitude));
  let bounds = L.latLngBounds(latLonList);
  if (latLonList.length != 0) {
    map.fitBounds(bounds);
  }
}


function put_markers_on_map(array, markers_only) {
  const n = array.length;
  let counter;
  let selected_icon;
  for (counter = 0; counter < n; counter++) {
    if (markers_only) {
        if (array[counter]["node_type"] === "consumer") {selected_icon= markerConsumer;}}
    else {
        if (array[counter]["node_type"] === "consumer") {
            if (array[counter]["is_connected"] === false) {selected_icon= markerShs;}
            else {selected_icon= markerConsumer;}
        }
        else {selected_icon= icons[array[counter]["node_type"]];}}
        L.marker([array[counter]["latitude"], array[counter]["longitude"]], {icon: selected_icon,})
            .on('click', markerOnClick).addTo(map);}
  zoomAll(map);
}