/**
 * The script integrates interactive drawing tools into a Leaflet.js map withina FastAPI web application. Primarily
 * utilized on the "consumer-selection" page, it empowers users to draw markers, rectangles, and polygons on the
 * map. These drawings are crucial for designating consumer locations and marking
 * specific areas of interest.
 *
 * Key Features:
 * - Polygon and Rectangle Drawing: Enables users to draw polygons and rectangles on the map with predefined
 *   shape and color settings, enhancing the map's interactivity and user engagement.
 *
 * - Custom Marker Controls: Facilitates the addition of distinct icons, such as power houses or consumers,
 *   onto the map, allowing for a more detailed and customized mapping experience.
 *
 * - Event Listeners: Actively listens for and responds to the creation of new shapes or markers. It ensures
 *   that the map elements are updated in real time, reflecting user actions instantaneously.
 *
 * - Dynamic Marker Management: Provides functionalities to dynamically add or remove markers based on user
 *   interactions, offering a flexible and responsive user interface.
 *
 * - Trash Bin Control: Incorporates a custom control to clear all drawn items from the map, simplifying the
 *   process of resetting or starting a new selection.
 *
 * - GeoSearch Integration: Utilizes the GeoSearch library to enable location searching capabilities,
 *   enhancing the usability and functionality of the map.
 *
 * - Consumer Toggle Feature: Includes a toggle mechanism to switch between adding and removing consumers
 *   from the map, offering versatility in map manipulation.
 *
 * - Consumer Count Display: Counts and displays the number of consumers currently present on the map, providing
 *   valuable insights at a glance.
 */


var polygonDrawer = new L.Draw.Polygon(map, {
    shapeOptions: {
        color: '#1F567D80'
    }
});

var rectangleDrawer = new L.Draw.Rectangle(map, {
    shapeOptions: {
        color: '#1F567D80'
    }
});


let isPowerHouseMarker = false

map.on('draw:created', function (e) {
    var layerType = e.layerType;
    if (layerType === 'marker' || layerType === 'rectangle' || layerType === 'polygon') {
        update_map_elements();
    }
});


// Add a listener that responds to the 'draw:created' event
map.on(L.Draw.Event.CREATED, function (event) {
        const layer = event.layer;

        if (event.layerType === 'marker') {
            const latLng = layer.getLatLng();
            const lat = latLng.lat;
            const lng = latLng.lng;


            if (isPowerHouseMarker) {
                let existingPowerHouseIndex = map_elements.findIndex(element => element.node_type == 'power-house');

                // If an element is found, remove it
                if (existingPowerHouseIndex !== -1) {
                    map_elements.splice(existingPowerHouseIndex, 1);
                    remove_marker_from_map();
                    put_markers_on_map(map_elements, true);
                }
                add_single_consumer_to_array(lat, lng, 'manual', 'power-house');
                drawMarker(lat, lng, 'power-house');
                isPowerHouseMarker = false;  // Reset the flag
            } else {
                add_single_consumer_to_array(lat, lng, 'manual', 'consumer');
                drawMarker(lat, lng, 'consumer');
                setTimeout(() => drawControl._toolbars.draw._modes.marker.handler.enable(), 100);
            }

            // Add a delay before re-enabling to bypass the default disable action

        } else {
            drawnItems.addLayer(layer);

            // Save the polygon coordinates in a variable
            polygonCoordinates.push(layer.getLatLngs());

            polygonDrawer.disable();
            if (!is_active) {
                add_buildings_inside_boundary({boundariesCoordinates: polygonCoordinates});
            } else {
                remove_buildings_inside_boundary({boundariesCoordinates: polygonCoordinates});
            }
            removeBoundaries();
        }
        ;
        count_consumers();
    }
)

var myCustomMarker = L.Icon.extend({
    options: {
        shadowUrl: null,
        iconAnchor: new L.Point(12, 12),
        iconSize: new L.Point(24, 24),
        iconUrl: "fastapi_app/files/public/media_files/icons/i_consumer.svg"
    }
});


const iconB = L.icon({
    iconUrl: "fastapi_app/files/public/media_files/icons/i_power_house.svg",
    iconSize: [12, 12], // size of the icon
    iconAnchor: [12, 12], // point of the icon which will correspond to marker's location
    popupAnchor: [1, -12] // point from which the popup should open relative to the iconAnchor
});


L.NewMarker = L.Draw.Marker.extend({
    options: {
        icon: iconB
    }
});


let drawControl = new L.Control.Draw({
    position: 'topleft',
    draw: {
        polyline: false,
        polygon: true,
        circle: false,
        circlemarker: false,
        rectangle: true,
        marker: {
            icon: new myCustomMarker
        }
    }
});

map.addControl(drawControl);

const CustomMarkerControl = L.Control.extend({
    options: {
        position: 'topleft'
    },

    onAdd: function (map) {
        const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
        L.DomEvent.disableClickPropagation(container);

        const link = L.DomUtil.create('a', 'leaflet-draw-draw-marker', container);
        link.href = '#';
        link.title = 'place power-house';

        // add an image inside the link
        const image = L.DomUtil.create('img', 'my-marker-icon', link);
        image.src = 'fastapi_app/files/public/media_files/icons/i_power_house_grey.svg';
        image.alt = 'Marker';
        image.style.width = '12px';
        image.style.height = '12px';

        L.DomEvent.on(link, 'click', L.DomEvent.stop)
            .on(link, 'click', function () {
                isPowerHouseMarker = true;

                // Disable any active drawing layer.
                for (let type in drawControl._toolbars.draw._modes) {
                    if (drawControl._toolbars.draw._modes[type].handler.enabled()) {
                        drawControl._toolbars.draw._modes[type].handler.disable();
                    }
                }

                new L.Draw.Marker(map, {icon: iconB}).enable();
            });


        return container;
    }
});

map.addControl(new CustomMarkerControl());


function add_single_consumer_to_array(latitude, longitude, how_added, node_type) {
    let consumer_type = 'household';
    if (node_type === 'power-house') {
        consumer_type = '';
    }

    map_elements.push({
        latitude: latitude,
        longitude: longitude,
        how_added: how_added,
        node_type: node_type,
        consumer_type: consumer_type,
        consumer_detail: 'default',
        custom_specification: '',
        shs_options: 0,
        is_connected: true
    });

    // Fetch the current value of n_consumers, increment it, and set it back
    let nConsumerElem = document.getElementById("n_consumers");
    let currentCount = parseInt(nConsumerElem.innerText, 10);
    nConsumerElem.innerText = currentCount + 1;
}


function remove_marker_from_map() {
    map.eachLayer((layer) => {
        if (layer instanceof L.Marker) {
            map.removeLayer(layer);
        }
    });
    document.getElementById("n_consumers").innerText = 0;
}

L.Control.Trashbin = L.Control.extend({
    options: {
        position: 'topleft',
    },

    onAdd: function () {
        const container = L.DomUtil.create('div', 'leaflet-control-trashbin leaflet-bar');
        const link = L.DomUtil.create('a', '', container);
        link.href = '#';
        link.title = 'Clear all';
        link.innerHTML = '🗑'; // Use the HTML entity for the trash bin icon (U+1F5D1)

        L.DomEvent.on(link, 'click', L.DomEvent.stopPropagation)
            .on(link, 'click', L.DomEvent.preventDefault)
            .on(link, 'click', customTrashBinAction);

        return container;
    },
});

function customTrashBinAction() {
    removeBoundaries();
    remove_marker_from_map();
    polygonCoordinates = [];
    map_elements = [];
}

const trashbinControl = new L.Control.Trashbin();
map.addControl(trashbinControl);


const searchProvider = new GeoSearch.OpenStreetMapProvider();

const searchControl = new GeoSearch.GeoSearchControl({
    provider: searchProvider,
    position: 'topleft',
    showMarker: false,
});

map.addControl(searchControl);

const searchInput = document.getElementById('search-input');

searchInput.addEventListener('keypress', async (event) => {
    if (event.key === 'Enter') {
        let query = searchInput.value;
        if (!query) return;

        let results = await searchProvider.search({query});

        // Retry if no results found or if the result is outside Nigeria without having 'Nigeria' in the query
        if ((!results || results.length === 0 || !isLatLngInMapBounds(results[0].y, results[0].x)) && !query.toLowerCase().includes("nigeria")) {
            query += ", Nigeria";
            results = await searchProvider.search({query});
        }

        if (results && results.length > 0) {
            const {x: lng, y: lat} = results[0];

            if (isLatLngInMapBounds(lat, lng)) {
                map.setView([lat, lng], 13);
            } else {
                const responseMsg = document.getElementById("responseMsg");
                responseMsg.innerHTML = 'Location is outside of Nigeria';
            }
        } else {
            alert('No results found');
        }
    }
});

function isLatLngInMapBounds(lat, lng) {
    const latLng = L.latLng(lat, lng);
    return map.options.maxBounds.contains(latLng);
}

let input = document.getElementById('toggleswitch');


function removeBoundaries() {
    drawnItems.clearLayers();
    polygonCoordinates = [];
}


var customControl = L.Control.extend({
    options: {
        position: 'bottomleft'
    },

    onAdd: function (map) {
        var container = L.DomUtil.create('div', 'my-custom-control');

        // Create the form
        var form = L.DomUtil.create('form', 'my-form', container);
        var label = L.DomUtil.create('label', '', form);
        label.textContent = 'Remove Consumers: ';
        var input = L.DomUtil.create('input', '', form);
        input.type = 'checkbox';

        // When the input changes, toggle your feature
        L.DomEvent.on(input, 'change', function () {
            is_active = !is_active;  // Toggle the is_active variable
            // TODO: Add your code here to do something with the map when the toggle changes
        });

        return container;
    }
});

// Add the control to the map
map.addControl(new customControl());


function unique_map_elements() {
    const uniqueLocations = new Set();
    const uniqueMapElements = [];
    for (let element of map_elements) {
        const locationKey = `${element.latitude},${element.longitude}`;
        if (!uniqueLocations.has(locationKey)) {
            uniqueLocations.add(locationKey);
            uniqueMapElements.push(element);
        }
    }
    map_elements = uniqueMapElements;
}

function count_consumers() {
    update_map_elements();
    unique_map_elements()
    const n = map_elements.length;
    let num_consumers = 0;  // Initialize the consumer counter
    for (let counter = 0; counter < n; counter++) {
        if (map_elements[counter]["node_type"] === "consumer") {
            num_consumers++;  // Increase the consumer counter
        }
    }
    document.getElementById("n_consumers").innerText = num_consumers;
}

