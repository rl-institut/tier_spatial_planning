// --------------------VARIABLES DECLARATION----------------------//

var markers = [];
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

var mainMap = L.map("leafletMap", {
  center: [11.3929, 9.1248], // 1st arg.: latitude, 2nd arg.: longitude
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
mainMap.addControl(search);

L.control.layers(osmMap, esriSatelliteMap).addTo(mainMap);

// custom zoom bar control that includes a Zoom Home function
mainMap.removeControl(mainMap.zoomControl);
L.Control.zoomHome = L.Control.extend({
  options: {
    position: "topleft",
    zoomInText: "+",
    zoomInTitle: "Zoom in",
    zoomOutText: "&#8722", //this is a long minus sign
    zoomOutTitle: "Zoom out",
    zoomHomeText:
      '<img src="fastapi_app/static/images/imgZoomToAll.png"></img>',
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
    var group = new L.featureGroup(markers).addTo(this._map);
    if (markers.length != 0) {
      this._map.fitBounds(group.getBounds());
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
zoomHome.addTo(mainMap);

function zoomAll(mainMap) {
  var group = new L.featureGroup(markers).addTo(mainMap);
  if (markers.length != 0) {
    mainMap.fitBounds(group.getBounds());
  }
}

L.easyButton(
  '<img class="leaflet-touch" src="fastapi_app/static/images/imgClearAll.png">',
  function (btn, map) {
    database_clear_all();
    position: "topleft";
  },
  "Clear all nodes from the map"
).addTo(mainMap);

var layer = L.geoJson(null).addTo(mainMap);

layer.fire("data:loading");
$.getJSON("http://server/path.geojson", function (data) {
  layer.fire("data:loaded");
  layer.addData(data);
});

var siteBoundaries = [];

var siteBoundaryLines = [];
var dashedBoundaryLine = null;

L.control.scale().addTo(mainMap);

var markerDefault = new L.Icon({
  iconUrl: "fastapi_app/static/images/markers/markerDefault.png",
  iconSize: [15, 15],
  iconAnchor: [7.5, 7.5],
  popupAnchor: [0, 0],
});

var markerHighDemand = new L.Icon({
  iconUrl: "fastapi_app/static/images/markers/markerHighDemand.png",
  iconSize: [18, 18],
});

var markerMediumDemand = new L.Icon({
  iconUrl: "fastapi_app/static/images/markers/markerMediumDemand.png",
  iconSize: [16, 16],
});

var markerLowDemand = new L.Icon({
  iconUrl: "fastapi_app/static/images/markers/markerLowDemand.png",
  iconSize: [14, 14],
});

var markerPole = new L.Icon({
  iconUrl: "fastapi_app/static/images/markers/markerPole.png",
  iconSize: [16, 16],
});

var markerShs = new L.Icon({
  iconUrl: "fastapi_app/static/images/markers/markerShs.png",
  iconSize: [16, 16],
});

var legend = L.control({ position: "bottomright" });
legend.onAdd = function (map) {
  var div = L.DomUtil.create("div", "info legend"),
    description = ["High Demand", "Medium Demand", "Low Demand", "Pole", "SHS"],
    image = [
      "fastapi_app/static/images/markers/markerHighDemand.png",
      "fastapi_app/static/images/markers/markerMediumDemand.png",
      "fastapi_app/static/images/markers/markerLowDemand.png",
      "fastapi_app/static/images/markers/markerPole.png",
      "fastapi_app/static/images/markers/markerShs.png",
    ];

  // loop through our density intervals and generate a label with a colored square for each interval
  for (var i = 0; i < description.length; i++) {
    div.innerHTML +=
      " <img src=" +
      image[i] +
      " height='20' width='20'>" +
      "&nbsp" +
      description[i] +
      "<br>";
  }
  return div;
};
legend.addTo(mainMap);

mainMap.on("click", function (e) {
  var poplocation = e.latlng;

  if (document.getElementById("radio_button_nodes_manually").checked) {
    if (document.getElementsByName("radio_button_nodes_manually")[0].checked) {
      database_add_from_js(
        {
          add_nodes: true,
          latitude: poplocation.lat,
          longitude: poplocation.lng,
          node_type: "high-demand",
          how_added: "manual"
        }
      );
      drawMarker(
        poplocation.lat,
        poplocation.lng,
        "high-demand"
      );
    }

    if (document.getElementsByName("radio_button_nodes_manually")[1].checked) {
      database_add_from_js(
        {
          add_nodes: true,
          latitude: poplocation.lat,
          longitude: poplocation.lng,
          node_type: "medium-demand"
        }
      );
      drawMarker(
        poplocation.lat,
        poplocation.lng,
        "medium-demand"
      );
    }

    if (document.getElementsByName("radio_button_nodes_manually")[2].checked) {
      database_add_from_js(
        {
          add_nodes: true,
          latitude: poplocation.lat,
          longitude: poplocation.lng,
          node_type: "low-demand"
        }
      );
      drawMarker(
        poplocation.lat,
        poplocation.lng,
        "low-demand"
      );
    }

    if (document.getElementsByName("radio_button_nodes_manually")[3].checked) {
      database_add_from_js(
        {
          add_nodes: true,
          latitude: poplocation.lat,
          longitude: poplocation.lng,
          node_type: "pole"
        }
      );
      drawMarker(
        poplocation.lat,
        poplocation.lng,
        "pole"
      );
    }
  }

  if (
    document.getElementById("radio_button_nodes_boundaries").checked &&
    (document.getElementById("button_draw_boundaries_add").innerHTML ===
      "Select" ||
      document.getElementById("button_draw_boundaries_remove").innerHTML ===
      "Remove")
  ) {
    siteBoundaries.push([poplocation.lat, poplocation.lng]);

    // adding the new solid line to siteBoundaryLines and draw it on the map
    siteBoundaryLines.push(L.polyline(siteBoundaries, { color: "black" }));
    siteBoundaryLines[siteBoundaryLines.length - 1].addTo(mainMap);

    // removing the dashed line
    if (dashedBoundaryLine) {
      mainMap.removeLayer(dashedBoundaryLine);
    }

    // creating a new dashed line closing the polygon
    dashedBoundaryLine = L.polyline(
      [siteBoundaries[0], siteBoundaries.slice(-1)[0]],
      { color: "black", dashArray: "10, 10", dashOffset: "20" }
    );

    // adding the new dashed line to the map
    dashedBoundaryLine.addTo(mainMap);
  }
});

// --------------------FUNCTIONS DECLARATION----------------------//

// INTERACTION WITH LEAFLET MAP

function drawMarker(latitude, longitude, type) {
  if (type === "high-demand") {
    icon_type = markerHighDemand;
  } else if (type === "medium-demand") {
    icon_type = markerMediumDemand;
  } else if (type === "low-demand") {
    icon_type = markerLowDemand;
  } else if (type === "pole") {
    icon_type = markerPole;
  } else if (type === "shs") {
    icon_type = markerShs;
  }
  markers.push(
    L.marker([latitude, longitude], { icon: icon_type }).addTo(mainMap)
  );
}

function drawLinkOnMap(
  latitude_from,
  longitude_from,
  latitude_to,
  longitude_to,
  color,
  map,
  weight,
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
    mainMap.removeLayer(line);
  }
  if (dashedBoundaryLine != null) {
    mainMap.removeLayer(dashedBoundaryLine);
  }
  siteBoundaries.length = 0;
  siteBoundaryLines.length = 0;
  dashedBoundaryLine = null;
}
