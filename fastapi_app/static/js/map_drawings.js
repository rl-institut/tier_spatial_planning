

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
  markerDrawer.enable();
    rectangleDrawer.disable();
  polygonDrawer.disable();
});

// Enable polygon drawing when the button is clicked
document.getElementById('drawPolygon').addEventListener('click', function () {
  markerDrawer.disable();
  rectangleDrawer.disable();
  polygonDrawer.enable();
});

document.getElementById('drawRectangle').addEventListener('click', function () {
  rectangleDrawer.enable();
    markerDrawer.disable();
    polygonDrawer.disable();
});

// Add a listener that responds to the 'draw:created' event
map.on(L.Draw.Event.CREATED, function (event) {
    const layer = event.layer;
    console.log('hello');
    console.log(event)
    if (event.layerType === 'marker') {
        const latLng = layer.getLatLng();
        const lat = latLng.lat;
        const lng = latLng.lng;
        add_single_consumer_to_array(lat, lng, 'manual', 'consumer')
        drawMarker(lat, lng,'consumer');
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




// Configure the drawing options
const drawControl = new L.Control.Draw({
    position: 'topleft',
    draw: {
        polyline: false,
        polygon: true,
        circle: false,
        circlemarker: false,
        marker: true,
        rectangle: true
    }
});

map.addControl(drawControl);


function add_single_consumer_to_array(latitude, longitude, how_added, node_type) {
  map_elements.push({
      latitude: latitude,
      longitude: longitude,
      how_added: how_added,
      node_type: node_type,
      surface_area: 0})
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
      responseMsg.innerHTML = 'Location is outside of Nigeria';
    } else {
        const responseMsg = document.getElementById("responseMsg");
        responseMsg.innerHTML = 'Location is outside of Nigeria';
        responseMsg.style.color = "red";
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
let is_active = false;
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