/**
 * This script is designed to create and manage an interactive map using Leaflet.js, tailored specifically for two
 * distinct pages within a FastAPI application: "consumer-selection" and "results".
 *
 * Features:
 * - On the "results" page, the script dynamically displays an array of markers, including consumers, enterprises,
 *   public services, powerhouses, poles, and solar home systems. Additionally, it visualizes links between these
 *   entities. All markers and links are automatically updated based on the data retrieved from the database.
 *
 * - The "consumer-selection" page, while also featuring these markers and links, goes a step further by incorporating
 *   drawing tools for enhanced user interaction. These drawing tools, however, are activated only when the
 *   "add_drawing_tools_to_map.js" script is also implemented.
 *
 * - The map is designed with multiple layers, including the standard OpenStreetMap and a Satellite view, allowing
 *   users to toggle between different map styles for better visualization.
 *
 * - Customizable map features include zoom control and a bespoke legend, specifically designed for a region with
 *   defined boundaries around Nigeria. This enhances the user experience by providing relevant geographical context
 *   and detail.
 *
 * Overall, this script is a key component in enhancing the interactivity and functionality of the map feature on
 * the FastAPI application, playing a crucial role in both the "consumer-selection" and "results" pages.
 */


var is_load_center = true;

const nigeriaBounds = [
    [4.2, 2.7], // Southwest corner
    [13.9, 14.7] // Northeast corner
];
let map;

var legend = L.control({position: "bottomright"});

let polygonCoordinates = [];

let map_elements = [];

var markerConsumer = new L.Icon({
    iconUrl: "fastapi_app/files/public/media_files/assets/icons/i_consumer.svg",
    iconSize: [18, 18],
});


var markerEnterprise = new L.Icon({
    iconUrl: "fastapi_app/files/public/media_files/assets/icons/i_enterprise.svg",
    iconSize: [18, 18],
});


var markerPublicservice = new L.Icon({
    iconUrl: "fastapi_app/files/public/media_files/icons/i_public_service.svg",
    iconSize: [18, 18],
});


var markerPowerHouse = new L.Icon({
    iconUrl: "fastapi_app/files/public/media_files/assets/icons/i_power_house.svg",
    iconSize: [12, 12],
});


var markerPole = new L.Icon({
    iconUrl: "fastapi_app/files/public/media_files/assets/icons/i_pole.svg",
    iconSize: [10, 10],
});


var markerShs = new L.Icon({
    iconUrl: "fastapi_app/files/public/media_files/assets/icons/i_shs.svg",
    iconSize: [16, 16],
});


var icons = {
    'consumer': markerConsumer,
    'power-house': markerPowerHouse,
    'pole': markerPole,
    'shs': markerShs,
};
var image = [
    "fastapi_app/files/public/media_files/icons/i_power_house.svg",
    "fastapi_app/files/public/media_files/icons/i_consumer.svg",
    "fastapi_app/files/public/media_files/icons/i_enterprise.svg",
    "fastapi_app/files/public/media_files/icons/i_public_service.svg",
    "fastapi_app/files/public/media_files/icons/i_pole.svg",
    "fastapi_app/files/public/media_files/assets/icons/i_shs.svg",
    "fastapi_app/files/public/media_files/assets/icons/i_distribution.svg",
    "fastapi_app/files/public/media_files/assets/icons/i_connection.svg",
];

const drawnItems = new L.FeatureGroup();

let is_active = false;

function initializeMap(center = null, zoom = null, bounds = null) {
    if (!map) {
        // Only initialize the map if it hasn't been initialized yet
        map = L.map('map', {
            preferCanvas: true, // This ensures Leaflet renders vectors and geometries on a Canvas.
            maxBounds: nigeriaBounds,
            maxBoundsViscosity: 1.0,
        });

        // Adjust map view based on the arguments provided
        if (center && zoom) {
            // Set the view using center and zoom if provided
            map.setView(center, zoom);
        } else if  (typeof bounds === 'object' && bounds !== null && Object.keys(bounds).length >= 4) {
            // Fit map to the given bounds if bounds are provided
            map.fitBounds(bounds);
        } else {
            // Fallback to a default view if no specific bounds or center/zoom are provided
            map.setView([9.8838, 5.9231], 6); // Default center and zoom
        }


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



        map.addLayer(drawnItems);

        var zoomAllControl = L.Control.extend({
            options: {
                position: 'topleft'
            },

            onAdd: function (map) {
                var container = L.DomUtil.create('div', 'leaflet-bar leaflet-control leaflet-control-custom');
                let baseUrl = window.location.protocol + "//" + window.location.hostname + (window.location.port ? ':' + window.location.port : '');
                let address = "url(" + baseUrl + "/fastapi_app/files/public/media_files/images/imgZoomToAll.png)"
                container.style.backgroundColor = 'white';
                container.style.backgroundImage = address;
                container.style.backgroundSize = "28px 28px";
                container.style.width = '32px';
                container.style.height = '32px';

                container.onclick = function () {
                    zoomAll(map);
                };

                return container;
            },
        });

        map.addControl(new zoomAllControl());
        load_legend();
        if (typeof loadDrawingToolsJS === 'function') {
            loadDrawingToolsJS();
        }
    }
}


function zoomAll(map) {
    let latLonList = map_elements.map(obj => L.latLng(obj.latitude, obj.longitude));
    let bounds = L.latLngBounds(latLonList);
    if (latLonList.length != 0) {
        map.fitBounds(bounds);
    }
}

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
    L.marker([latitude, longitude], {icon: icon_type}).on('click', markerOnClick).addTo(map)
}

async function put_markers_on_map(array, markers_only) {
    const n = array.length;
    let counter;
    let selected_icon;

    // Initialize the counters
    let num_consumers = 0;
    let num_households = 0;
    let num_enterprises = 0;
    let num_public_services = 0;

    let latLonList = array.map(obj => L.latLng(obj.latitude, obj.longitude));
    let bounds = L.latLngBounds(latLonList);

    initializeMap(null, null, bounds);

    for (counter = 0; counter < n; counter++) {
        if (array[counter]["node_type"] === "consumer") {
            num_consumers++;  // Increase the consumer counter

            // Count the specific types of consumers
            if (array[counter]["consumer_type"] === "household") {
                num_households++;
            } else if (array[counter]["consumer_type"] === "enterprise") {
                num_enterprises++;
            } else if (array[counter]["consumer_type"] === "public_service") {
                num_public_services++;
            }

            // Determine the icon to use
            if (markers_only) {
                if (array[counter]["shs_options"] == 2) {
                    selected_icon = markerShs;
                } else if (array[counter]["consumer_type"] === "household") {
                    selected_icon = markerConsumer;
                } else if (array[counter]["consumer_type"] === "enterprise") {
                    selected_icon = markerEnterprise;
                } else if (array[counter]["consumer_type"] === "public_service") {
                    selected_icon = markerPublicservice;
                }
            } else {
                if (array[counter]["is_connected"] === false) {
                    selected_icon = markerShs;
                } else if (array[counter]["consumer_type"] === "household") {
                    selected_icon = markerConsumer;
                } else if (array[counter]["consumer_type"] === "enterprise") {
                    selected_icon = markerEnterprise;
                } else if (array[counter]["consumer_type"] === "public_service") {
                    selected_icon = markerPublicservice;
                }
            }
        } else if (markers_only) {
            selected_icon = markerPowerHouse;
        } else {
            selected_icon = icons[array[counter]["node_type"]];
        }

        // Add the marker to the map
        L.marker([array[counter]["latitude"], array[counter]["longitude"]], {icon: selected_icon})
            .on('click', markerOnClick)
            .addTo(map);
    }

    // Update the elements with the counts
    if (document.getElementById("n_consumers")) {
        document.getElementById("n_consumers").innerText = num_consumers;
    }
    if (document.getElementById("n_households")) {
        document.getElementById("n_households").innerText = num_households;
    }
    if (document.getElementById("n_enterprises")) {
        document.getElementById("n_enterprises").innerText = num_enterprises;
    }
    if (document.getElementById("n_public_services")) {
        document.getElementById("n_public_services").innerText = num_public_services;
    }

    zoomAll(map);
    if (typeof loadDrawingToolsJS === 'undefined' || loadDrawingToolsJS === null) {
        db_links_to_js(project_id);
    }
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

function markerOnClick(e) {
    if (is_active) {
        L.DomEvent.stopPropagation(e);
        map_elements = map_elements.filter(function (obj) {
            return obj.latitude !== e.latlng.lng && obj.longitude !== e.latlng.lat;
        });
        map.eachLayer(function (layer) {
            if (layer instanceof L.Marker) {
                let markerLatLng = layer.getLatLng();
                if (markerLatLng.lat === e.latlng.lat && markerLatLng.lng === e.latlng.lng) {
                    map.removeLayer(layer);
                }
            }
        });
    }
}

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

// Function to load external script dynamically

