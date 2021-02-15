var mainMap = L.map('leafletMap').setView([51.505, -0.09], 2);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        tileSize: 512,
        zoomOffset: -1,
        minZoom: 1,
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        crossOrigin: true,
      }).addTo(mainMap);

mainMap.on('click', function(e) {
  var poplocation = e.latlng;
  console.log(poplocation)
  L.marker([e.latlng.lat, e.latlng.lng]).addTo(mainMap)
});



