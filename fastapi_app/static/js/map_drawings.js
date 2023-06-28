// Create a polygon draw handler object
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

var markerDrawer = new L.Draw.Marker(map);
document.getElementById('drawMarker').addEventListener('click', function () {
    update_map_elements();
  markerDrawer.enable();
    rectangleDrawer.disable();
  polygonDrawer.disable();
});

// Enable polygon drawing when the button is clicked
document.getElementById('drawPolygon').addEventListener('click', function () {
    update_map_elements();
  markerDrawer.disable();
  rectangleDrawer.disable();
  polygonDrawer.enable();
});

let isPowerHouseMarker = false

map.on('draw:created', function (e) {
    var layerType = e.layerType;
    if (layerType === 'marker' || layerType === 'rectangle' || layerType === 'polygon') {
        update_map_elements();
    }
});

document.getElementById('drawRectangle').addEventListener('click', function () {
    update_map_elements();
    rectangleDrawer.enable();
    markerDrawer.disable();
    polygonDrawer.disable();
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
        }
         else {
                add_single_consumer_to_array(lat, lng, 'manual', 'consumer');
                drawMarker(lat, lng, 'consumer');
                setTimeout(() => drawControl._toolbars.draw._modes.marker.handler.enable(), 100);
        }

        // Add a delay before re-enabling to bypass the default disable action

    }
    else {
        drawnItems.addLayer(layer);

        // Save the polygon coordinates in a variable
        polygonCoordinates.push(layer.getLatLngs());

        polygonDrawer.disable();
        if (!is_active)
        {
            add_buildings_inside_boundary({boundariesCoordinates: polygonCoordinates });
        }
        else
        {
            remove_buildings_inside_boundary({boundariesCoordinates: polygonCoordinates });
        }
        removeBoundaries();
    }
});

var myCustomMarker = L.Icon.extend({
    options: {
        shadowUrl: null,
        iconAnchor: new L.Point(12, 12),
        iconSize: new L.Point(24, 24),
        iconUrl: "fastapi_app/static/assets/icons/i_consumer.svg"
    }
});


const iconB = L.icon({
    iconUrl: "fastapi_app/static/assets/icons/i_power_house.svg",
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

    onAdd: function(map) {
        const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
        L.DomEvent.disableClickPropagation(container);

        const link = L.DomUtil.create('a', 'leaflet-draw-draw-marker', container);
        link.href = '#';
        link.title = 'place power-house';

        // add an image inside the link
        const image = L.DomUtil.create('img', 'my-marker-icon', link);
        image.src = 'fastapi_app/static/assets/icons/i_power_house_grey.svg';
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

                new L.Draw.Marker(map, { icon: iconB }).enable();
            });


        return container;
    }
});

map.addControl(new CustomMarkerControl());


function add_single_consumer_to_array(latitude, longitude, how_added, node_type) {
  let consumer_type = 'household';
  if (node_type === 'power-house') {consumer_type = ''}
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
  })
}


function remove_marker_from_map() {
  map.eachLayer((layer) => {
    if (layer instanceof L.Marker) {
      map.removeLayer(layer);
    }
  });
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
    const query = searchInput.value;
    if (query) {
      const results = await searchProvider.search({ query });
      if (results && results.length > 0) {
        const { x: lng, y: lat } = results[0];

        if (isLatLngInMapBounds(lat, lng)) {
      map.setView([lat, lng], 13);
    } else {
        const responseMsg = document.getElementById("responseMsg");
        responseMsg.innerHTML = 'Location is outside of Nigeria';
      }} else {
        alert('No results found');
      }
  }
}});

function isLatLngInMapBounds(lat, lng) {
  const latLng = L.latLng(lat, lng);
  return map.options.maxBounds.contains(latLng);
}

let input = document.getElementById('toggleswitch');

input.addEventListener('change',function(){
    if(this.checked) {
        is_active = true;
        }
    else {
        is_active = false;
        }
    });


function removeBoundaries() {
        drawnItems.clearLayers();
        polygonCoordinates = [];
}





