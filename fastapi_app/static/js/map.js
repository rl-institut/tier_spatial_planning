// --------------------VARIABLES DECLARATION----------------------//

var markers = [];
var lines = [];
siteGeojson = "";

var osmLayer = L.tileLayer(
  "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  {
    tileSize: 512,
    zoomOffset: -1,
    minZoom: 1,
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

var mainMap = L.map("leafletMap", {
  center: [11.3929, 9.1248],
  zoom: 17,
  layers: [osmLayer],
});

L.control.layers(osmMap, esriSatelliteMap).addTo(mainMap);

var mapClickEvent = "add_node";

var siteBoundaries = [];

var siteBoundaryLines = [];
var dashedBoundaryLine = null;

L.control.scale().addTo(mainMap);

var householdMarker = new L.Icon({
  iconUrl: "fastapi_app/static/images/markers/marker-household.png",
  iconSize: [10, 10],
  iconAnchor: [5, 5],
  popupAnchor: [0, 0],
});

var hubMarker = new L.Icon({
  iconUrl: "fastapi_app/static/images/markers/marker-hub.png",
  iconSize: [14, 14],
  iconAnchor: [7, 7],
  popupAnchor: [0, 0],
});

var shsMarker = new L.Icon({
  iconUrl: "fastapi_app/static/images/markers/marker-shs.png",
  iconSize: [10, 10],
  iconAnchor: [5, 5],
  popupAnchor: [0, 0],
});

mainMap.on("click", function (e) {
  var poplocation = e.latlng;

  if (mapClickEvent == "add_node") {
    if (document.getElementsByName("radio_button_new_node_type")[0].checked) {
      addNodeToDatBase(
        poplocation.lat,
        poplocation.lng,
        "undefinded",
        false,
        default_household_required_capacity,
        default_household_max_power
      );
      drawDefaultMarker(poplocation.lat, poplocation.lng);
    }

    if (document.getElementsByName("radio_button_new_node_type")[1].checked) {
      addNodeToDatBase(
        poplocation.lat,
        poplocation.lng,
        "household",
        true,
        default_household_required_capacity,
        default_household_max_power
      );
      drawHouseholdMarker(poplocation.lat, poplocation.lng);
    }

    if (document.getElementsByName("radio_button_new_node_type")[2].checked) {
      addNodeToDatBase(
        poplocation.lat,
        poplocation.lng,
        "meterhub",
        true,
        2 * default_household_required_capacity,
        2 * default_household_max_power
      );
      drawMeterhubMarker(poplocation.lat, poplocation.lng);
    }
  }

  if (mapClickEvent == "draw_boundaries") {
    siteBoundaries.push([poplocation.lat, poplocation.lng]);

    // add new solid line to siteBoundaryLines and draw it on map
    siteBoundaryLines.push(L.polyline(siteBoundaries, { color: "black" }));

    siteBoundaryLines[siteBoundaryLines.length - 1].addTo(mainMap);
    // Remove dashed line
    if (dashedBoundaryLine) {
      mainMap.removeLayer(dashedBoundaryLine);
    }

    // Create new dashed line closing the polygon
    dashedBoundaryLine = L.polyline(
      [siteBoundaries[0], siteBoundaries.slice(-1)[0]],
      { color: "black", dashArray: "10, 10", dashOffset: "20" }
    );

    // Add new dashed line to map
    dashedBoundaryLine.addTo(mainMap);
  }
});

// --------------------FUNCTIONS DECLARATION----------------------//

// INTERACTION WITH LEAFLET MAP

function drawDefaultMarker(latitude, longitude) {
  markers.push(L.marker([latitude, longitude]).addTo(mainMap));
}

function drawMeterhubMarker(latitude, longitude) {
  markers.push(
    L.marker([latitude, longitude], { icon: hubMarker }).addTo(mainMap)
  );
}

function drawHouseholdMarker(latitude, longitude) {
  markers.push(
    L.marker([latitude, longitude], { icon: householdMarker }).addTo(mainMap)
  );
}

function drawLinkOnMap(
  latitude_from,
  longitude_from,
  latitude_to,
  longitude_to,
  color,
  map,
  weight = 3,
  opacity = 0.5
) {
  var pointA = new L.LatLng(latitude_from, longitude_from);
  var pointB = new L.LatLng(latitude_to, longitude_to);
  var pointList = [pointA, pointB];

  var link_polyline = new L.polyline(pointList, {
    color: color,
    weight: weight,
    opacity: 0.5,
    smoothFactor: 1,
  });
  lines.push(link_polyline.addTo(map));
}

function ereaseLinksFromMap(map) {
  for (line of lines) {
    map.removeLayer(line);
  }
  lines.length = 0;
}

function removeBoundaries() {
  // Remove all boundary lines and markers
  for (line of siteBoundaryLines) {
    mainMap.removeLayer(line);
  }
  if (dashedBoundaryLine != null) {
    mainMap.removeLayer(dashedBoundaryLine);
  }
  siteBoundaries.length = 0;
  siteBoundaryLines.length = 0;
  dashedBoundaryLine = null;
}
