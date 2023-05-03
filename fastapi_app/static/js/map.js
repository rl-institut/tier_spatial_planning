// --------------------VARIABLES DECLARATION----------------------//

let map_elements =[];

var lines = [];

var osmLayer = L.tileLayer(
  "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  {
    tileSize: 512,
    zoomOffset: -1,
    minZoom: 1,
    maxZoom: 19,
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    crossOrigin: true,
  }
);

var osmMap = {
  osmBaseMap: osmLayer,
};

var esriWorldImageryLayer = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  {
    tileSize: 512,
    zoomOffset: -1,
    minZoom: 1,
    maxZoom: 18,
    attribution:
      "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
  }
);

var esriSatelliteMap = {
  esriBaseMap: esriWorldImageryLayer,
};

lat = 9.8838;
lon = 5.9231;

var map = L.map("leafletMap", {
  center: [lat, lon], // 1st arg.: latitude, 2nd arg.: longitude
  zoom: 17,
  layers: [osmLayer],
});

// add the search box to the map
const provider = new GeoSearch.OpenStreetMapProvider();
const search = new GeoSearch.GeoSearchControl({
  provider: new GeoSearch.OpenStreetMapProvider(),
  position: "topleft",
  style: "button",
  searchLabel: "Enter a Location...",
  autoComplete: true,
  updateMap: true,
  showMarker: false,
  autoClose: true,
  keepResult: true,
});
map.addControl(search);

L.control.layers(osmMap, esriSatelliteMap).addTo(map);

// custom zoom bar control that includes a Zoom Home function
map.removeControl(map.zoomControl);
L.Control.zoomHome = L.Control.extend({
  options: {
    position: "topleft",
    // zoomInText: "+",
    zoomInText:'<img class="leaflet-zoom-in-out" src="/fastapi_app/static/assets/icons/i_zoom_in.svg"></img>',
    zoomInTitle: "Zoom in",
    // zoomOutText: "&#8722", //this is a long minus sign
    zoomOutText:'<img class="leaflet-zoom-in-out" src="/fastapi_app/static/assets/icons/i_zoom_out.svg"></img>',
    zoomOutTitle: "Zoom out",
    zoomHomeText:
      '<img class="leaflet-zoom-fit" src="/fastapi_app/static/assets/icons/i_zoom_fit.svg"></img>',
    zoomHomeTitle: "Show all nodes",
  },

  onAdd: function (map) {
    var controlName = "gin-control-zoom",
      container = L.DomUtil.create("div", controlName + " leaflet-bar"),
      options = this.options;

    this._zoomInButton = this._createButton(
      options.zoomInText,
      options.zoomInTitle,
      controlName + "-in",
      container,
      this._zoomIn
    );
    this._zoomHomeButton = this._createButton(
      options.zoomHomeText,
      options.zoomHomeTitle,
      controlName + "-home",
      container,
      this._zoomHome
    );
    this._zoomOutButton = this._createButton(
      options.zoomOutText,
      options.zoomOutTitle,
      controlName + "-out",
      container,
      this._zoomOut
    );

    this._updateDisabled();
    map.on("zoomend zoomlevelschange", this._updateDisabled, this);

    return container;
  },

  onRemove: function (map) {
    map.off("zoomend zoomlevelschange", this._updateDisabled, this);
  },

  _zoomIn: function (e) {
    this._map.zoomIn(e.shiftKey ? 3 : 1);
  },

  _zoomOut: function (e) {
    this._map.zoomOut(e.shiftKey ? 3 : 1);
  },

  _zoomHome: function (e) {
    let latLonList = map_elements.map(obj => L.latLng(obj.lat, obj.lon));
    let bounds = L.latLngBounds(latLonList);
    if (map_elements.length != 0) {
      this._map.fitBounds(bounds);
    }
  },

  _createButton: function (html, title, className, container, fn) {
    var link = L.DomUtil.create("a", className, container);
    link.innerHTML = html;
    link.href = "#";
    link.title = title;

    L.DomEvent.on(link, "mousedown dblclick", L.DomEvent.stopPropagation)
      .on(link, "click", L.DomEvent.stop)
      .on(link, "click", fn, this)
      .on(link, "click", this._refocusOnMap, this);

    return link;
  },

  _updateDisabled: function () {
    var map = this._map,
      className = "leaflet-disabled";

    L.DomUtil.removeClass(this._zoomInButton, className);
    L.DomUtil.removeClass(this._zoomOutButton, className);

    if (map._zoom === map.getMinZoom()) {
      L.DomUtil.addClass(this._zoomOutButton, className);
    }
    if (map._zoom === map.getMaxZoom()) {
      L.DomUtil.addClass(this._zoomInButton, className);
    }
  },
});

// add the new control to the map
var zoomHome = new L.Control.zoomHome();
zoomHome.addTo(map);

function zoomAll(mainMap) {
  let latLonList = map_elements.map(obj => L.latLng(obj.latitude, obj.longitude));
  let bounds = L.latLngBounds(latLonList);
  if (latLonList.length != 0) {
    mainMap.fitBounds(bounds);
  }
}

L.easyButton(
  // '<img class="leaflet-touch" src="'+src_clear+'">',
  '<img class="leaflet-touch" src="/fastapi_app/static/assets/icons/i_clear_all.svg">',
  function (btn, map) {
    const urlParams = new URLSearchParams(window.location.search);
    project_id = urlParams.get('project_id');
  remove_marker_from_map();
  map_elements = [];
  removeBoundaries();
    position: "topleft";
  },
  "Clear all nodes from the map"
).addTo(map);

var layer = L.geoJson(null).addTo(map);

// layer.fire("data:loading");
// $.getJSON("http://server/path.geojson", function (data) {
//   layer.fire("data:loaded");
//   layer.addData(data);
// });

let siteBoundaries = [];

var siteBoundaryLines = [];
var dashedBoundaryLine = null;

L.control.scale().addTo(map);

var markerConsumer = new L.Icon({
  iconUrl: "fastapi_app/static/assets/icons/i_consumer.svg",
  iconSize: [12, 12],
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

var icons = {
  'consumer': markerConsumer,
  'power-house': markerPowerHouse,
  'pole': markerPole,
  'shs': markerShs,
};

var legend = L.control({ position: "bottomright" });
legend.onAdd = function (map) {
  var div = L.DomUtil.create("div", "info legend"),
    description = ["Power House", "Consumer", "Pole", "SHS", "Distribution", "Connection"],
    image = [
      "fastapi_app/static/assets/icons/i_power_house.svg",
      "fastapi_app/static/assets/icons/i_consumer.svg",
      "fastapi_app/static/assets/icons/i_pole.svg",
      "fastapi_app/static/assets/icons/i_shs.svg",
      "fastapi_app/static/assets/icons/i_distribution.svg",
      "fastapi_app/static/assets/icons/i_connection.svg",
    ];

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

map.on("click", function (e) {
  var poplocation = e.latlng;

  if (document.getElementById("selectionMap").checked) {
    add_single_consumer_to_array(poplocation.lat, poplocation.lng, 'manual', 'consumer')
    drawMarker(poplocation.lat, poplocation.lng,'consumer');}
  if (
    document.getElementById("selectionBoundaries").checked &&
    (document.getElementById("btnDrawBoundariesAdd").innerText === 'Draw Lines' ||
      document.getElementById("btnDrawBoundariesRemove").innerText === 'Draw Lines')
  ) {

    let polylineBounds = new L.LatLngBounds([poplocation.lat, poplocation.lng]);
    siteBoundaries.push([poplocation.lat, poplocation.lng]);
    // adding the new solid line to siteBoundaryLines and draw it on the map
    siteBoundaryLines.push(L.polyline(siteBoundaries, { color: "black" , bounds: polylineBounds}));
    siteBoundaryLines[siteBoundaryLines.length - 1].addTo(map);
    // removing the dashed line
    if (dashedBoundaryLine) {
      map.removeLayer(dashedBoundaryLine);
    }
    // creating a new dashed line closing the polygon
    dashedBoundaryLine = L.polyline(
      [siteBoundaries[0], siteBoundaries.slice(-1)[0]],
      { color: "black", dashArray: "10, 10", dashOffset: "20" }
    );
    // adding the new dashed line to the map
    dashedBoundaryLine.addTo(map);
  };
});



// --------------------FUNCTIONS DECLARATION----------------------//

// INTERACTION WITH LEAFLET MAP

function remove_marker_from_map() {
  map.eachLayer((layer) => {
    if (layer instanceof L.Marker) {
      map.removeLayer(layer);
    }
  });
}

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

function markerOnClick(e)
{ L.DomEvent.stopPropagation(e);
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
  lines.push(
    link_polyline.bindTooltip(
      pointA.distanceTo(pointB).toFixed(2).toString() + " m"
    ).addTo(map));
}

function removeLinksFromMap(map) {
  for (line of lines) {
    map.removeLayer(line);
  }
  lines.length = 0;
}

function removeBoundaries() {
  // Remove all boundary lines and markers
  for (line of siteBoundaryLines) {
    map.removeLayer(line);
  }
  if (dashedBoundaryLine != null) {
    map.removeLayer(dashedBoundaryLine);
  }
  siteBoundaries.length = 0;
  siteBoundaryLines.length = 0;
  dashedBoundaryLine = null;
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
        console.log(selected_icon);
        L.marker([array[counter]["latitude"], array[counter]["longitude"]], {icon: selected_icon,})
            .on('click', markerOnClick).addTo(map);}
  zoomAll(map);
}

function add_single_consumer_to_array(latitude, longitude, how_added, node_type) {
  map_elements.push({
      latitude: latitude,
      longitude: longitude,
      how_added: how_added,
      node_type: node_type,
      surface_area: 0})
}

/************************************************************/
/*                    BOUNDARY SELECTION                    */
/************************************************************/
// selecting boundaries of the site for adding new nodes


function boundary_select(mode, project_id) {
    button_text = 'Start'
    if (mode == 'add') {
        button_class = 'btn--success'
        var btnAddRemove = document.getElementById("btnDrawBoundariesAdd");
    } else {
        var btnAddRemove = document.getElementById("btnDrawBoundariesRemove");
        button_class = 'btn--error'
    }
    // changing the label of the button
    if (btnAddRemove.innerText === button_text) {
        btnAddRemove.innerText = 'Draw Lines';
        btnAddRemove.classList.toggle(button_class)
    } else {
        btnAddRemove.innerText = button_text;
        btnAddRemove.classList.remove(button_class)
    }
    // only when a boundary is drawn, the next steps will be executed
    if (siteBoundaryLines.length > 0) {
        if (mode == 'add') {
            add_buildings_inside_boundary({boundariesCoordinates: siteBoundaries });
        }
        else {
            remove_buildings_inside_boundary({boundariesCoordinates: siteBoundaries });
        }
            removeBoundaries();
    }
}

