// Create the map and set the center and zoom level

var is_load_center = true;

const nigeriaBounds = [
  [4.2, 2.7], // Southwest corner
  [13.9, 14.7] // Northeast corner
];


const map = L.map('map', {
  center: [9.8838, 5.9231],
  zoom: 6,
  maxBounds: nigeriaBounds,
  maxBoundsViscosity: 1.0,
});

let is_active = false;

// Define the OSM layer
let osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
});

// Define the Esri satellite layer
let satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri'
});

// Add the OSM layer to the map as the default
osmLayer.addTo(map);

// Define the base layers for the control
let baseMaps = {
    "OpenStreetMap": osmLayer,
    "Satellite": satelliteLayer
};

// Add the layer control to the map
L.control.layers(baseMaps).addTo(map);


const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);
let polygonCoordinates = [];
let map_elements =[];


var markerConsumer = new L.Icon({
  iconUrl: "fastapi_app/static/assets/icons/i_consumer.svg",
  iconSize: [18, 18],
});


var markerEnterprise = new L.Icon({
  iconUrl: "fastapi_app/static/assets/icons/i_enterprise.svg",
  iconSize: [18, 18],
});


var markerPublicservice = new L.Icon({
  iconUrl: "fastapi_app/static/assets/icons/i_public_service.svg",
  iconSize: [18, 18],
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
  iconUrl: "fastapi_app/static/assets/icons/i_shs.svg",
  iconSize: [16, 16],
});


function drawMarker(latitude, longitude, type) {
  if (type === "consumer") {
    icon_type = markerConsumer;
  } else if (type === "pole") {
    icon_type = markerPole;
  } else if (type === "shs") {
    icon_type = markerShs;
  } else if (type === "power-house") {
    icon_type = markerPowerHouse;
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
  if (latLonList.length != 0) {map.fitBounds(bounds);}}


function put_markers_on_map(array, markers_only) {
    const n = array.length;
    let counter;
    let selected_icon;

    // Initialize the consumer counter
    let num_consumers = 0;

    for (counter = 0; counter < n; counter++) {
        if (array[counter]["node_type"] === "consumer") {
            num_consumers++;  // Increase the consumer counter

            if (markers_only) {
                if (array[counter]["shs_options"] == 2) {selected_icon = markerShs;}
                else if (array[counter]["consumer_type"] === "household") {selected_icon = markerConsumer;}
                else if (array[counter]["consumer_type"] === "enterprise") {selected_icon = markerEnterprise;}
                else if (array[counter]["consumer_type"] === "public_service") {selected_icon = markerPublicservice;}
            } else {
                if (array[counter]["is_connected"] === false) {selected_icon = markerShs;}
                else if (array[counter]["consumer_type"] === "household") {selected_icon = markerConsumer;}
                else if (array[counter]["consumer_type"] === "enterprise") {selected_icon = markerEnterprise;}
                else if (array[counter]["consumer_type"] === "public_service") {selected_icon = markerPublicservice;}
            }
        } else if (markers_only) {
            selected_icon = markerPowerHouse;
        } else {
            selected_icon = icons[array[counter]["node_type"]];
        }

        L.marker([array[counter]["latitude"], array[counter]["longitude"]], {icon: selected_icon,})
            .on('click', markerOnClick).addTo(map);
    }
    // Update the element with the count of consumers
    if (document.getElementById("n_consumers")) {
    document.getElementById("n_consumers").innerText = num_consumers;
    }
    zoomAll(map);
}



function removeLinksFromMap(map) {
  for (line of polygonCoordinates) {
    map.removeLayer(line);
  }
  polygonCoordinates.length = 0;
}

function put_links_on_map(links) {
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
}

function markerOnClick(e)
{ if (is_active) {
    L.DomEvent.stopPropagation(e);
    map_elements = map_elements.filter(function (obj) {
        return obj.latitude !== e.latlng.lng && obj.longitude !== e.latlng.lat;});
  map.eachLayer(function (layer) {
  if (layer instanceof L.Marker) {
    let markerLatLng = layer.getLatLng();
    if (markerLatLng.lat === e.latlng.lat && markerLatLng.lng === e.latlng.lng) {
      map.removeLayer(layer);
    }
  }
});
}}

function drawLinkOnMap(
  latitude_from,
  longitude_from,
  latitude_to,
  longitude_to,
  color,
  map,
  weight,
  opacity,
) {
  var pointA = new L.LatLng(latitude_from, longitude_from);
  var pointB = new L.LatLng(latitude_to, longitude_to);
  var pointList = [pointA, pointB];

  var link_polyline = new L.polyline(pointList, {
    color: color,
    weight: weight,
    opacity: opacity,
    smoothFactor: 1,
  });
  polygonCoordinates.push(
    link_polyline.bindTooltip(
      pointA.distanceTo(pointB).toFixed(2).toString() + " m"
    ).addTo(map));
}

var zoomAllControl = L.Control.extend({
    options: {
        position: 'topleft'
    },

    onAdd: function (map) {
        var container = L.DomUtil.create('div', 'leaflet-bar leaflet-control leaflet-control-custom');
        let baseUrl = window.location.protocol + "//" + window.location.hostname + (window.location.port ? ':' + window.location.port: '');
        let address = "url(" + baseUrl + "/fastapi_app/static/images/imgZoomToAll.png)"
        container.style.backgroundColor = 'white';
        container.style.backgroundImage = address;
        container.style.backgroundSize = "28px 28px";
        container.style.width = '32px';
        container.style.height = '32px';

        container.onclick = function(){
            zoomAll(map);
        };

        return container;
    },
});

map.addControl(new zoomAllControl());



var image = [
      "fastapi_app/static/assets/icons/i_power_house.svg",
      "fastapi_app/static/assets/icons/i_consumer.svg",
      "fastapi_app/static/assets/icons/i_enterprise.svg",
      "fastapi_app/static/assets/icons/i_public_service.svg",
      "fastapi_app/static/assets/icons/i_pole.svg",
      "fastapi_app/static/assets/icons/i_shs.svg",
      "fastapi_app/static/assets/icons/i_distribution.svg",
      "fastapi_app/static/assets/icons/i_connection.svg",
];

var legend = L.control({ position: "bottomright" });

function load_legend() {
    // Obtain the page name, for example using window.location.pathname
    // Replace it with your own logic for getting the page name
        // If there's already a legend, remove it
    if (legend) {
        map.removeControl(legend);
    }
    var pageName = window.location.pathname;

    var description = ["Load Center", "Household", "Enterprise", "Public Service", "Pole", "Solar Home System", "Distribution", "Connection"];

    if (pageName === "/simulation_results" && is_load_center === false) {
        description[0] = "Power House";
    }
    // Add the legend

    legend.onAdd = function (map) {
      var div = L.DomUtil.create("div", "info legend");

      // loop through our density intervals and generate a label with a colored square for each interval
      for (var i = 0; i < description.length; i++) {
        div.innerHTML +=
          " <img src=" +
          image[i] +
          " height='12' width='12'>" +
          "&nbsp" +
          description[i] +
          "<br>";
      }
      return div;
    };
    legend.addTo(map);
}
load_legend();
