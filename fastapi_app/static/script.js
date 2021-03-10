var mainMap = L.map("leafletMap").setView([9.07798, 7.704826], 5);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  tileSize: 512,
  zoomOffset: -1,
  minZoom: 1,
  attribution:
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  crossOrigin: true,
}).addTo(mainMap);

var mapClickEvent = "add_default_node";

var siteBoundaries = [];

var siteBoundaryLines = [];
var dashedBoundaryLine = null;

siteGeojson = "";

L.control.scale().addTo(mainMap);
var geojsonFeature = {
  type: "FeatureCollection",
  generator: "overpass-ide",
  copyright:
    "The data included in this document is from www.openstreetmap.org. The data is made available under ODbL.",
  timestamp: "2021-03-10T12:06:57Z",
  features: [
    {
      type: "Feature",
      properties: {
        "@id": "way/734320192",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324443, 11.391193],
            [9.1324593, 11.3911677],
            [9.132478, 11.3911784],
            [9.1324629, 11.3912037],
            [9.1324443, 11.391193],
          ],
        ],
      },
      id: "way/734320192",
    },
    {
      type: "Feature",
      properties: {
        "@id": "way/734320193",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324312, 11.3911858],
            [9.1324369, 11.3911874],
            [9.1324351, 11.3911932],
            [9.1324294, 11.3911916],
            [9.1324312, 11.3911858],
          ],
        ],
      },
      id: "way/734320193",
    },
    {
      type: "Feature",
      properties: {
        "@id": "way/734320204",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1323759, 11.3908055],
            [9.1324523, 11.3908292],
            [9.1324344, 11.3908849],
            [9.1323579, 11.3908612],
            [9.1323759, 11.3908055],
          ],
        ],
      },
      id: "way/734320204",
    },
    {
      type: "Feature",
      properties: {
        "@id": "way/734320205",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324107, 11.3908049],
            [9.1324409, 11.3907187],
            [9.1324922, 11.390736],
            [9.132462, 11.3908221],
            [9.1324107, 11.3908049],
          ],
        ],
      },
      id: "way/734320205",
    },
    {
      type: "Feature",
      properties: {
        "@id": "way/734320206",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324671, 11.3909081],
            [9.1325147, 11.3908193],
            [9.1325572, 11.3908413],
            [9.1325096, 11.39093],
            [9.1324671, 11.3909081],
          ],
        ],
      },
      id: "way/734320206",
    },
    {
      type: "Feature",
      properties: {
        "@id": "way/734320207",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324825, 11.390979],
            [9.132507, 11.3909363],
            [9.1325559, 11.3909633],
            [9.1325314, 11.391006],
            [9.1324825, 11.390979],
          ],
        ],
      },
      id: "way/734320207",
    },
    {
      type: "Feature",
      properties: {
        "@id": "way/734320208",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324409, 11.3909311],
            [9.132476, 11.3909441],
            [9.1324592, 11.3909875],
            [9.1324241, 11.3909744],
            [9.1324409, 11.3909311],
          ],
        ],
      },
      id: "way/734320208",
    },
    {
      type: "Feature",
      properties: {
        "@id": "way/734320209",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1321646, 11.3908936],
            [9.1322275, 11.3908491],
            [9.1322657, 11.3909011],
            [9.1322029, 11.3909455],
            [9.1321646, 11.3908936],
          ],
        ],
      },
      id: "way/734320209",
    },
    {
      type: "Feature",
      properties: {
        "@id": "way/734320210",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324925, 11.3910954],
            [9.1325227, 11.3910389],
            [9.1325488, 11.3910523],
            [9.1325187, 11.3911088],
            [9.1324925, 11.3910954],
          ],
        ],
      },
      id: "way/734320210",
    },
    {
      type: "Feature",
      properties: {
        "@id": "way/734320211",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324195, 11.3911427],
            [9.1324408, 11.3910914],
            [9.1324951, 11.3911131],
            [9.1324738, 11.3911644],
            [9.1324195, 11.3911427],
          ],
        ],
      },
      id: "way/734320211",
    },
    {
      type: "Feature",
      properties: {
        "@id": "way/734320212",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1323001, 11.3912177],
            [9.1323269, 11.3911741],
            [9.1323692, 11.391199],
            [9.1323423, 11.3912426],
            [9.1323001, 11.3912177],
          ],
        ],
      },
      id: "way/734320212",
    },
    {
      type: "Feature",
      properties: {
        "@id": "way/734320213",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1323779, 11.3912703],
            [9.1324029, 11.3912315],
            [9.1324612, 11.3912676],
            [9.1324362, 11.3913064],
            [9.1323779, 11.3912703],
          ],
        ],
      },
      id: "way/734320213",
    },
    {
      type: "Feature",
      properties: {
        "@id": "way/734320214",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1322216, 11.3912985],
            [9.1322371, 11.3912723],
            [9.1322673, 11.3912894],
            [9.1322518, 11.3913156],
            [9.1322216, 11.3912985],
          ],
        ],
      },
      id: "way/734320214",
    },
    {
      type: "Feature",
      properties: {
        "@id": "way/734320215",
        building: "yes",
      },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1321278, 11.3912887],
            [9.132166, 11.3912194],
            [9.1322143, 11.391245],
            [9.132176, 11.3913143],
            [9.1321278, 11.3912887],
          ],
        ],
      },
      id: "way/734320215",
    },
  ],
};

var geojsonFeature1 = {
  version: 0.6,
  generator: "Overpass API 0.7.56.9 76e5016d",
  osm3s: {
    timestamp_osm_base: "2021-03-10T11:54:42Z",
    copyright:
      "The data included in this document is from www.openstreetmap.org. The data is made available under ODbL.",
  },
  elements: [
    {
      type: "way",
      id: 734320192,
      nodes: [6876753977, 6876753974, 6876753975, 6876753976, 6876753977],
      tags: {
        building: "yes",
      },
    },
    {
      type: "way",
      id: 734320193,
      nodes: [6876753981, 6876753978, 6876753979, 6876753980, 6876753981],
      tags: {
        building: "yes",
      },
    },
    {
      type: "way",
      id: 734320204,
      nodes: [6876754025, 6876754022, 6876754023, 6876754024, 6876754025],
      tags: {
        building: "yes",
      },
    },
    {
      type: "way",
      id: 734320205,
      nodes: [6876754029, 6876754026, 6876754027, 6876754028, 6876754029],
      tags: {
        building: "yes",
      },
    },
    {
      type: "way",
      id: 734320206,
      nodes: [6876754033, 6876754030, 6876754031, 6876754032, 6876754033],
      tags: {
        building: "yes",
      },
    },
    {
      type: "way",
      id: 734320207,
      nodes: [6876754037, 6876754036, 6876754035, 6876754034, 6876754037],
      tags: {
        building: "yes",
      },
    },
    {
      type: "way",
      id: 734320208,
      nodes: [6876754041, 6876754040, 6876754039, 6876754038, 6876754041],
      tags: {
        building: "yes",
      },
    },
    {
      type: "way",
      id: 734320209,
      nodes: [6876754045, 6876754044, 6876754043, 6876754042, 6876754045],
      tags: {
        building: "yes",
      },
    },
    {
      type: "way",
      id: 734320210,
      nodes: [6876754049, 6876754046, 6876754047, 6876754048, 6876754049],
      tags: {
        building: "yes",
      },
    },
    {
      type: "way",
      id: 734320211,
      nodes: [6876754053, 6876754052, 6876754051, 6876754050, 6876754053],
      tags: {
        building: "yes",
      },
    },
    {
      type: "way",
      id: 734320212,
      nodes: [6876754057, 6876754056, 6876754055, 6876754054, 6876754057],
      tags: {
        building: "yes",
      },
    },
    {
      type: "way",
      id: 734320213,
      nodes: [6876754061, 6876754060, 6876754059, 6876754058, 6876754061],
      tags: {
        building: "yes",
      },
    },
    {
      type: "way",
      id: 734320214,
      nodes: [6876754065, 6876754064, 6876754063, 6876754062, 6876754065],
      tags: {
        building: "yes",
      },
    },
    {
      type: "way",
      id: 734320215,
      nodes: [6876754069, 6876754068, 6876754067, 6876754066, 6876754069],
      tags: {
        building: "yes",
      },
    },
    {
      type: "node",
      id: 6876753974,
      lat: 11.3912037,
      lon: 9.1324629,
    },
    {
      type: "node",
      id: 6876753975,
      lat: 11.3911784,
      lon: 9.132478,
    },
    {
      type: "node",
      id: 6876753976,
      lat: 11.3911677,
      lon: 9.1324593,
    },
    {
      type: "node",
      id: 6876753977,
      lat: 11.391193,
      lon: 9.1324443,
    },
    {
      type: "node",
      id: 6876753978,
      lat: 11.3911916,
      lon: 9.1324294,
    },
    {
      type: "node",
      id: 6876753979,
      lat: 11.3911932,
      lon: 9.1324351,
    },
    {
      type: "node",
      id: 6876753980,
      lat: 11.3911874,
      lon: 9.1324369,
    },
    {
      type: "node",
      id: 6876753981,
      lat: 11.3911858,
      lon: 9.1324312,
    },
    {
      type: "node",
      id: 6876754022,
      lat: 11.3908612,
      lon: 9.1323579,
    },
    {
      type: "node",
      id: 6876754023,
      lat: 11.3908849,
      lon: 9.1324344,
    },
    {
      type: "node",
      id: 6876754024,
      lat: 11.3908292,
      lon: 9.1324523,
    },
    {
      type: "node",
      id: 6876754025,
      lat: 11.3908055,
      lon: 9.1323759,
    },
    {
      type: "node",
      id: 6876754026,
      lat: 11.3908221,
      lon: 9.132462,
    },
    {
      type: "node",
      id: 6876754027,
      lat: 11.390736,
      lon: 9.1324922,
    },
    {
      type: "node",
      id: 6876754028,
      lat: 11.3907187,
      lon: 9.1324409,
    },
    {
      type: "node",
      id: 6876754029,
      lat: 11.3908049,
      lon: 9.1324107,
    },
    {
      type: "node",
      id: 6876754030,
      lat: 11.39093,
      lon: 9.1325096,
    },
    {
      type: "node",
      id: 6876754031,
      lat: 11.3908413,
      lon: 9.1325572,
    },
    {
      type: "node",
      id: 6876754032,
      lat: 11.3908193,
      lon: 9.1325147,
    },
    {
      type: "node",
      id: 6876754033,
      lat: 11.3909081,
      lon: 9.1324671,
    },
    {
      type: "node",
      id: 6876754034,
      lat: 11.3909363,
      lon: 9.132507,
    },
    {
      type: "node",
      id: 6876754035,
      lat: 11.3909633,
      lon: 9.1325559,
    },
    {
      type: "node",
      id: 6876754036,
      lat: 11.391006,
      lon: 9.1325314,
    },
    {
      type: "node",
      id: 6876754037,
      lat: 11.390979,
      lon: 9.1324825,
    },
    {
      type: "node",
      id: 6876754038,
      lat: 11.3909441,
      lon: 9.132476,
    },
    {
      type: "node",
      id: 6876754039,
      lat: 11.3909875,
      lon: 9.1324592,
    },
    {
      type: "node",
      id: 6876754040,
      lat: 11.3909744,
      lon: 9.1324241,
    },
    {
      type: "node",
      id: 6876754041,
      lat: 11.3909311,
      lon: 9.1324409,
    },
    {
      type: "node",
      id: 6876754042,
      lat: 11.3908491,
      lon: 9.1322275,
    },
    {
      type: "node",
      id: 6876754043,
      lat: 11.3909011,
      lon: 9.1322657,
    },
    {
      type: "node",
      id: 6876754044,
      lat: 11.3909455,
      lon: 9.1322029,
    },
    {
      type: "node",
      id: 6876754045,
      lat: 11.3908936,
      lon: 9.1321646,
    },
    {
      type: "node",
      id: 6876754046,
      lat: 11.3911088,
      lon: 9.1325187,
    },
    {
      type: "node",
      id: 6876754047,
      lat: 11.3910523,
      lon: 9.1325488,
    },
    {
      type: "node",
      id: 6876754048,
      lat: 11.3910389,
      lon: 9.1325227,
    },
    {
      type: "node",
      id: 6876754049,
      lat: 11.3910954,
      lon: 9.1324925,
    },
    {
      type: "node",
      id: 6876754050,
      lat: 11.3910914,
      lon: 9.1324408,
    },
    {
      type: "node",
      id: 6876754051,
      lat: 11.3911131,
      lon: 9.1324951,
    },
    {
      type: "node",
      id: 6876754052,
      lat: 11.3911644,
      lon: 9.1324738,
    },
    {
      type: "node",
      id: 6876754053,
      lat: 11.3911427,
      lon: 9.1324195,
    },
    {
      type: "node",
      id: 6876754054,
      lat: 11.3911741,
      lon: 9.1323269,
    },
    {
      type: "node",
      id: 6876754055,
      lat: 11.391199,
      lon: 9.1323692,
    },
    {
      type: "node",
      id: 6876754056,
      lat: 11.3912426,
      lon: 9.1323423,
    },
    {
      type: "node",
      id: 6876754057,
      lat: 11.3912177,
      lon: 9.1323001,
    },
    {
      type: "node",
      id: 6876754058,
      lat: 11.3912315,
      lon: 9.1324029,
    },
    {
      type: "node",
      id: 6876754059,
      lat: 11.3912676,
      lon: 9.1324612,
    },
    {
      type: "node",
      id: 6876754060,
      lat: 11.3913064,
      lon: 9.1324362,
    },
    {
      type: "node",
      id: 6876754061,
      lat: 11.3912703,
      lon: 9.1323779,
    },
    {
      type: "node",
      id: 6876754062,
      lat: 11.3912723,
      lon: 9.1322371,
    },
    {
      type: "node",
      id: 6876754063,
      lat: 11.3912894,
      lon: 9.1322673,
    },
    {
      type: "node",
      id: 6876754064,
      lat: 11.3913156,
      lon: 9.1322518,
    },
    {
      type: "node",
      id: 6876754065,
      lat: 11.3912985,
      lon: 9.1322216,
    },
    {
      type: "node",
      id: 6876754066,
      lat: 11.3912194,
      lon: 9.132166,
    },
    {
      type: "node",
      id: 6876754067,
      lat: 11.391245,
      lon: 9.1322143,
    },
    {
      type: "node",
      id: 6876754068,
      lat: 11.3913143,
      lon: 9.132176,
    },
    {
      type: "node",
      id: 6876754069,
      lat: 11.3912887,
      lon: 9.1321278,
    },
  ],
};

var geojsonFeature2 = {
  type: "FeatureCollection",
  features: [
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324629, 11.3912037] },
      properties: {
        type: "node",
        id: 6876753974,
        lat: 11.3912037,
        lon: 9.1324629,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.132478, 11.3911784] },
      properties: {
        type: "node",
        id: 6876753975,
        lat: 11.3911784,
        lon: 9.132478,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324593, 11.3911677] },
      properties: {
        type: "node",
        id: 6876753976,
        lat: 11.3911677,
        lon: 9.1324593,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324443, 11.391193] },
      properties: {
        type: "node",
        id: 6876753977,
        lat: 11.391193,
        lon: 9.1324443,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324294, 11.3911916] },
      properties: {
        type: "node",
        id: 6876753978,
        lat: 11.3911916,
        lon: 9.1324294,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324351, 11.3911932] },
      properties: {
        type: "node",
        id: 6876753979,
        lat: 11.3911932,
        lon: 9.1324351,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324369, 11.3911874] },
      properties: {
        type: "node",
        id: 6876753980,
        lat: 11.3911874,
        lon: 9.1324369,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324312, 11.3911858] },
      properties: {
        type: "node",
        id: 6876753981,
        lat: 11.3911858,
        lon: 9.1324312,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1330146, 11.3915466] },
      properties: {
        type: "node",
        id: 6876753982,
        lat: 11.3915466,
        lon: 9.1330146,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1330119, 11.3915742] },
      properties: {
        type: "node",
        id: 6876753983,
        lat: 11.3915742,
        lon: 9.1330119,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1329814, 11.3915713] },
      properties: {
        type: "node",
        id: 6876753984,
        lat: 11.3915713,
        lon: 9.1329814,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1329841, 11.3915437] },
      properties: {
        type: "node",
        id: 6876753985,
        lat: 11.3915437,
        lon: 9.1329841,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335531, 11.3914161] },
      properties: {
        type: "node",
        id: 6876753998,
        lat: 11.3914161,
        lon: 9.1335531,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335685, 11.3914155] },
      properties: {
        type: "node",
        id: 6876753999,
        lat: 11.3914155,
        lon: 9.1335685,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335694, 11.3914359] },
      properties: {
        type: "node",
        id: 6876754000,
        lat: 11.3914359,
        lon: 9.1335694,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.133554, 11.3914366] },
      properties: {
        type: "node",
        id: 6876754001,
        lat: 11.3914366,
        lon: 9.133554,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.133513, 11.3914191] },
      properties: {
        type: "node",
        id: 6876754002,
        lat: 11.3914191,
        lon: 9.133513,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335331, 11.3914204] },
      properties: {
        type: "node",
        id: 6876754003,
        lat: 11.3914204,
        lon: 9.1335331,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335319, 11.3914385] },
      properties: {
        type: "node",
        id: 6876754004,
        lat: 11.3914385,
        lon: 9.1335319,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335118, 11.3914372] },
      properties: {
        type: "node",
        id: 6876754005,
        lat: 11.3914372,
        lon: 9.1335118,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1334772, 11.3914765] },
      properties: {
        type: "node",
        id: 6876754006,
        lat: 11.3914765,
        lon: 9.1334772,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335054, 11.3914779] },
      properties: {
        type: "node",
        id: 6876754007,
        lat: 11.3914779,
        lon: 9.1335054,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335031, 11.3915253] },
      properties: {
        type: "node",
        id: 6876754008,
        lat: 11.3915253,
        lon: 9.1335031,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1334749, 11.391524] },
      properties: {
        type: "node",
        id: 6876754009,
        lat: 11.391524,
        lon: 9.1334749,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1336004, 11.391484] },
      properties: {
        type: "node",
        id: 6876754010,
        lat: 11.391484,
        lon: 9.1336004,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1336594, 11.3914932] },
      properties: {
        type: "node",
        id: 6876754011,
        lat: 11.3914932,
        lon: 9.1336594,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1336526, 11.3915352] },
      properties: {
        type: "node",
        id: 6876754012,
        lat: 11.3915352,
        lon: 9.1336526,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335936, 11.391526] },
      properties: {
        type: "node",
        id: 6876754013,
        lat: 11.391526,
        lon: 9.1335936,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1336144, 11.3915613] },
      properties: {
        type: "node",
        id: 6876754014,
        lat: 11.3915613,
        lon: 9.1336144,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1337029, 11.3915718] },
      properties: {
        type: "node",
        id: 6876754015,
        lat: 11.3915718,
        lon: 9.1337029,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1336982, 11.3916101] },
      properties: {
        type: "node",
        id: 6876754016,
        lat: 11.3916101,
        lon: 9.1336982,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1336097, 11.3915996] },
      properties: {
        type: "node",
        id: 6876754017,
        lat: 11.3915996,
        lon: 9.1336097,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1323579, 11.3908612] },
      properties: {
        type: "node",
        id: 6876754022,
        lat: 11.3908612,
        lon: 9.1323579,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324344, 11.3908849] },
      properties: {
        type: "node",
        id: 6876754023,
        lat: 11.3908849,
        lon: 9.1324344,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324523, 11.3908292] },
      properties: {
        type: "node",
        id: 6876754024,
        lat: 11.3908292,
        lon: 9.1324523,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1323759, 11.3908055] },
      properties: {
        type: "node",
        id: 6876754025,
        lat: 11.3908055,
        lon: 9.1323759,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.132462, 11.3908221] },
      properties: {
        type: "node",
        id: 6876754026,
        lat: 11.3908221,
        lon: 9.132462,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324922, 11.390736] },
      properties: {
        type: "node",
        id: 6876754027,
        lat: 11.390736,
        lon: 9.1324922,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324409, 11.3907187] },
      properties: {
        type: "node",
        id: 6876754028,
        lat: 11.3907187,
        lon: 9.1324409,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324107, 11.3908049] },
      properties: {
        type: "node",
        id: 6876754029,
        lat: 11.3908049,
        lon: 9.1324107,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1325096, 11.39093] },
      properties: {
        type: "node",
        id: 6876754030,
        lat: 11.39093,
        lon: 9.1325096,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1325572, 11.3908413] },
      properties: {
        type: "node",
        id: 6876754031,
        lat: 11.3908413,
        lon: 9.1325572,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1325147, 11.3908193] },
      properties: {
        type: "node",
        id: 6876754032,
        lat: 11.3908193,
        lon: 9.1325147,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324671, 11.3909081] },
      properties: {
        type: "node",
        id: 6876754033,
        lat: 11.3909081,
        lon: 9.1324671,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.132507, 11.3909363] },
      properties: {
        type: "node",
        id: 6876754034,
        lat: 11.3909363,
        lon: 9.132507,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1325559, 11.3909633] },
      properties: {
        type: "node",
        id: 6876754035,
        lat: 11.3909633,
        lon: 9.1325559,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1325314, 11.391006] },
      properties: {
        type: "node",
        id: 6876754036,
        lat: 11.391006,
        lon: 9.1325314,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324825, 11.390979] },
      properties: {
        type: "node",
        id: 6876754037,
        lat: 11.390979,
        lon: 9.1324825,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.132476, 11.3909441] },
      properties: {
        type: "node",
        id: 6876754038,
        lat: 11.3909441,
        lon: 9.132476,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324592, 11.3909875] },
      properties: {
        type: "node",
        id: 6876754039,
        lat: 11.3909875,
        lon: 9.1324592,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324241, 11.3909744] },
      properties: {
        type: "node",
        id: 6876754040,
        lat: 11.3909744,
        lon: 9.1324241,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324409, 11.3909311] },
      properties: {
        type: "node",
        id: 6876754041,
        lat: 11.3909311,
        lon: 9.1324409,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1322275, 11.3908491] },
      properties: {
        type: "node",
        id: 6876754042,
        lat: 11.3908491,
        lon: 9.1322275,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1322657, 11.3909011] },
      properties: {
        type: "node",
        id: 6876754043,
        lat: 11.3909011,
        lon: 9.1322657,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1322029, 11.3909455] },
      properties: {
        type: "node",
        id: 6876754044,
        lat: 11.3909455,
        lon: 9.1322029,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1321646, 11.3908936] },
      properties: {
        type: "node",
        id: 6876754045,
        lat: 11.3908936,
        lon: 9.1321646,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1325187, 11.3911088] },
      properties: {
        type: "node",
        id: 6876754046,
        lat: 11.3911088,
        lon: 9.1325187,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1325488, 11.3910523] },
      properties: {
        type: "node",
        id: 6876754047,
        lat: 11.3910523,
        lon: 9.1325488,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1325227, 11.3910389] },
      properties: {
        type: "node",
        id: 6876754048,
        lat: 11.3910389,
        lon: 9.1325227,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324925, 11.3910954] },
      properties: {
        type: "node",
        id: 6876754049,
        lat: 11.3910954,
        lon: 9.1324925,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324408, 11.3910914] },
      properties: {
        type: "node",
        id: 6876754050,
        lat: 11.3910914,
        lon: 9.1324408,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324951, 11.3911131] },
      properties: {
        type: "node",
        id: 6876754051,
        lat: 11.3911131,
        lon: 9.1324951,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324738, 11.3911644] },
      properties: {
        type: "node",
        id: 6876754052,
        lat: 11.3911644,
        lon: 9.1324738,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324195, 11.3911427] },
      properties: {
        type: "node",
        id: 6876754053,
        lat: 11.3911427,
        lon: 9.1324195,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1323269, 11.3911741] },
      properties: {
        type: "node",
        id: 6876754054,
        lat: 11.3911741,
        lon: 9.1323269,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1323692, 11.391199] },
      properties: {
        type: "node",
        id: 6876754055,
        lat: 11.391199,
        lon: 9.1323692,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1323423, 11.3912426] },
      properties: {
        type: "node",
        id: 6876754056,
        lat: 11.3912426,
        lon: 9.1323423,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1323001, 11.3912177] },
      properties: {
        type: "node",
        id: 6876754057,
        lat: 11.3912177,
        lon: 9.1323001,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324029, 11.3912315] },
      properties: {
        type: "node",
        id: 6876754058,
        lat: 11.3912315,
        lon: 9.1324029,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324612, 11.3912676] },
      properties: {
        type: "node",
        id: 6876754059,
        lat: 11.3912676,
        lon: 9.1324612,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1324362, 11.3913064] },
      properties: {
        type: "node",
        id: 6876754060,
        lat: 11.3913064,
        lon: 9.1324362,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1323779, 11.3912703] },
      properties: {
        type: "node",
        id: 6876754061,
        lat: 11.3912703,
        lon: 9.1323779,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1322371, 11.3912723] },
      properties: {
        type: "node",
        id: 6876754062,
        lat: 11.3912723,
        lon: 9.1322371,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1322673, 11.3912894] },
      properties: {
        type: "node",
        id: 6876754063,
        lat: 11.3912894,
        lon: 9.1322673,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1322518, 11.3913156] },
      properties: {
        type: "node",
        id: 6876754064,
        lat: 11.3913156,
        lon: 9.1322518,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1322216, 11.3912985] },
      properties: {
        type: "node",
        id: 6876754065,
        lat: 11.3912985,
        lon: 9.1322216,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.132166, 11.3912194] },
      properties: {
        type: "node",
        id: 6876754066,
        lat: 11.3912194,
        lon: 9.132166,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1322143, 11.391245] },
      properties: {
        type: "node",
        id: 6876754067,
        lat: 11.391245,
        lon: 9.1322143,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.132176, 11.3913143] },
      properties: {
        type: "node",
        id: 6876754068,
        lat: 11.3913143,
        lon: 9.132176,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1321278, 11.3912887] },
      properties: {
        type: "node",
        id: 6876754069,
        lat: 11.3912887,
        lon: 9.1321278,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1336193, 11.3916283] },
      properties: {
        type: "node",
        id: 6876754090,
        lat: 11.3916283,
        lon: 9.1336193,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1336086, 11.3916796] },
      properties: {
        type: "node",
        id: 6876754091,
        lat: 11.3916796,
        lon: 9.1336086,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335507, 11.3916679] },
      properties: {
        type: "node",
        id: 6876754092,
        lat: 11.3916679,
        lon: 9.1335507,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335614, 11.3916167] },
      properties: {
        type: "node",
        id: 6876754093,
        lat: 11.3916167,
        lon: 9.1335614,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1334351, 11.3916497] },
      properties: {
        type: "node",
        id: 6876754094,
        lat: 11.3916497,
        lon: 9.1334351,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1333386, 11.3916431] },
      properties: {
        type: "node",
        id: 6876754095,
        lat: 11.3916431,
        lon: 9.1333386,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1333415, 11.3916022] },
      properties: {
        type: "node",
        id: 6876754096,
        lat: 11.3916022,
        lon: 9.1333415,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.133438, 11.3916088] },
      properties: {
        type: "node",
        id: 6876754097,
        lat: 11.3916088,
        lon: 9.133438,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1334704, 11.3915378] },
      properties: {
        type: "node",
        id: 6876754098,
        lat: 11.3915378,
        lon: 9.1334704,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335321, 11.3915365] },
      properties: {
        type: "node",
        id: 6876754099,
        lat: 11.3915365,
        lon: 9.1335321,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335332, 11.3915864] },
      properties: {
        type: "node",
        id: 6876754100,
        lat: 11.3915864,
        lon: 9.1335332,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1334715, 11.3915877] },
      properties: {
        type: "node",
        id: 6876754101,
        lat: 11.3915877,
        lon: 9.1334715,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1334414, 11.3913629] },
      properties: {
        type: "node",
        id: 6876754102,
        lat: 11.3913629,
        lon: 9.1334414,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335125, 11.3913668] },
      properties: {
        type: "node",
        id: 6876754103,
        lat: 11.3913668,
        lon: 9.1335125,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1335104, 11.3914024] },
      properties: {
        type: "node",
        id: 6876754104,
        lat: 11.3914024,
        lon: 9.1335104,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1334394, 11.3913984] },
      properties: {
        type: "node",
        id: 6876754105,
        lat: 11.3913984,
        lon: 9.1334394,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1333345, 11.3913937] },
      properties: {
        type: "node",
        id: 6876754106,
        lat: 11.3913937,
        lon: 9.1333345,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1333278, 11.3914634] },
      properties: {
        type: "node",
        id: 6876754107,
        lat: 11.3914634,
        lon: 9.1333278,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.133265, 11.3914576] },
      properties: {
        type: "node",
        id: 6876754108,
        lat: 11.3914576,
        lon: 9.133265,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1332717, 11.3913879] },
      properties: {
        type: "node",
        id: 6876754109,
        lat: 11.3913879,
        lon: 9.1332717,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1331434, 11.3914043] },
      properties: {
        type: "node",
        id: 6876754110,
        lat: 11.3914043,
        lon: 9.1331434,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.133181, 11.3914122] },
      properties: {
        type: "node",
        id: 6876754111,
        lat: 11.3914122,
        lon: 9.133181,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1331725, 11.391451] },
      properties: {
        type: "node",
        id: 6876754112,
        lat: 11.391451,
        lon: 9.1331725,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1331349, 11.3914431] },
      properties: {
        type: "node",
        id: 6876754113,
        lat: 11.3914431,
        lon: 9.1331349,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1328605, 11.3913682] },
      properties: {
        type: "node",
        id: 6876754114,
        lat: 11.3913682,
        lon: 9.1328605,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1329316, 11.3913827] },
      properties: {
        type: "node",
        id: 6876754115,
        lat: 11.3913827,
        lon: 9.1329316,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1329405, 11.3913406] },
      properties: {
        type: "node",
        id: 6876754116,
        lat: 11.3913406,
        lon: 9.1329405,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1328694, 11.3913261] },
      properties: {
        type: "node",
        id: 6876754117,
        lat: 11.3913261,
        lon: 9.1328694,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1329251, 11.3914072] },
      properties: {
        type: "node",
        id: 6876754118,
        lat: 11.3914072,
        lon: 9.1329251,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1329934, 11.3914243] },
      properties: {
        type: "node",
        id: 6876754119,
        lat: 11.3914243,
        lon: 9.1329934,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.132982, 11.3914681] },
      properties: {
        type: "node",
        id: 6876754120,
        lat: 11.3914681,
        lon: 9.132982,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1329136, 11.391451] },
      properties: {
        type: "node",
        id: 6876754121,
        lat: 11.391451,
        lon: 9.1329136,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1328451, 11.3914785] },
      properties: {
        type: "node",
        id: 6876754122,
        lat: 11.3914785,
        lon: 9.1328451,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1328666, 11.3913838] },
      properties: {
        type: "node",
        id: 6876754123,
        lat: 11.3913838,
        lon: 9.1328666,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1328251, 11.3913748] },
      properties: {
        type: "node",
        id: 6876754124,
        lat: 11.3913748,
        lon: 9.1328251,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1328037, 11.3914694] },
      properties: {
        type: "node",
        id: 6876754125,
        lat: 11.3914694,
        lon: 9.1328037,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1332219, 11.3916141] },
      properties: {
        type: "node",
        id: 6876754126,
        lat: 11.3916141,
        lon: 9.1332219,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.133226, 11.3915444] },
      properties: {
        type: "node",
        id: 6876754127,
        lat: 11.3915444,
        lon: 9.133226,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1331778, 11.3915417] },
      properties: {
        type: "node",
        id: 6876754128,
        lat: 11.3915417,
        lon: 9.1331778,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1331738, 11.3916114] },
      properties: {
        type: "node",
        id: 6876754129,
        lat: 11.3916114,
        lon: 9.1331738,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1330545, 11.3915982] },
      properties: {
        type: "node",
        id: 6876754130,
        lat: 11.3915982,
        lon: 9.1330545,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1331403, 11.3916061] },
      properties: {
        type: "node",
        id: 6876754131,
        lat: 11.3916061,
        lon: 9.1331403,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1331363, 11.3916482] },
      properties: {
        type: "node",
        id: 6876754132,
        lat: 11.3916482,
        lon: 9.1331363,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1330504, 11.3916403] },
      properties: {
        type: "node",
        id: 6876754133,
        lat: 11.3916403,
        lon: 9.1330504,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.132768, 11.3916077] },
      properties: {
        type: "node",
        id: 6876754134,
        lat: 11.3916077,
        lon: 9.132768,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.132819, 11.3916327] },
      properties: {
        type: "node",
        id: 6876754135,
        lat: 11.3916327,
        lon: 9.132819,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1327983, 11.3916732] },
      properties: {
        type: "node",
        id: 6876754136,
        lat: 11.3916732,
        lon: 9.1327983,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1327474, 11.3916482] },
      properties: {
        type: "node",
        id: 6876754137,
        lat: 11.3916482,
        lon: 9.1327474,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1329432, 11.3916836] },
      properties: {
        type: "node",
        id: 6876754138,
        lat: 11.3916836,
        lon: 9.1329432,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1329714, 11.3916191] },
      properties: {
        type: "node",
        id: 6876754139,
        lat: 11.3916191,
        lon: 9.1329714,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1329311, 11.3916022] },
      properties: {
        type: "node",
        id: 6876754140,
        lat: 11.3916022,
        lon: 9.1329311,
      },
    },
    {
      type: "Feature",
      geometry: { type: "Point", coordinates: [9.1329029, 11.3916666] },
      properties: {
        type: "node",
        id: 6876754141,
        lat: 11.3916666,
        lon: 9.1329029,
      },
    },
  ],
};

var geojsonFeature3 = {
  type: "FeatureCollection",
  generator: "overpass-ide, formated by PeopleSun WP4 Tool",
  timestamp: "2021-03-10 17:09:52",
  features: [
    {
      type: "Feature",
      property: { "@id": "way/734320192", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324443, 11.391193],
            [9.1324629, 11.3912037],
            [9.132478, 11.3911784],
            [9.1324593, 11.3911677],
            [9.1324443, 11.391193],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320193", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324312, 11.3911858],
            [9.1324294, 11.3911916],
            [9.1324351, 11.3911932],
            [9.1324369, 11.3911874],
            [9.1324312, 11.3911858],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320194", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1329841, 11.3915437],
            [9.1329814, 11.3915713],
            [9.1330119, 11.3915742],
            [9.1330146, 11.3915466],
            [9.1329841, 11.3915437],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320198", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.133554, 11.3914366],
            [9.1335694, 11.3914359],
            [9.1335685, 11.3914155],
            [9.1335531, 11.3914161],
            [9.133554, 11.3914366],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320199", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1335118, 11.3914372],
            [9.1335319, 11.3914385],
            [9.1335331, 11.3914204],
            [9.133513, 11.3914191],
            [9.1335118, 11.3914372],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320200", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1334749, 11.391524],
            [9.1335031, 11.3915253],
            [9.1335054, 11.3914779],
            [9.1334772, 11.3914765],
            [9.1334749, 11.391524],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320201", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1335936, 11.391526],
            [9.1336526, 11.3915352],
            [9.1336594, 11.3914932],
            [9.1336004, 11.391484],
            [9.1335936, 11.391526],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320202", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1336097, 11.3915996],
            [9.1336982, 11.3916101],
            [9.1337029, 11.3915718],
            [9.1336144, 11.3915613],
            [9.1336097, 11.3915996],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320204", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1323759, 11.3908055],
            [9.1323579, 11.3908612],
            [9.1324344, 11.3908849],
            [9.1324523, 11.3908292],
            [9.1323759, 11.3908055],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320205", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324107, 11.3908049],
            [9.132462, 11.3908221],
            [9.1324922, 11.390736],
            [9.1324409, 11.3907187],
            [9.1324107, 11.3908049],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320206", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324671, 11.3909081],
            [9.1325096, 11.39093],
            [9.1325572, 11.3908413],
            [9.1325147, 11.3908193],
            [9.1324671, 11.3909081],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320207", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324825, 11.390979],
            [9.1325314, 11.391006],
            [9.1325559, 11.3909633],
            [9.132507, 11.3909363],
            [9.1324825, 11.390979],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320208", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324409, 11.3909311],
            [9.1324241, 11.3909744],
            [9.1324592, 11.3909875],
            [9.132476, 11.3909441],
            [9.1324409, 11.3909311],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320209", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1321646, 11.3908936],
            [9.1322029, 11.3909455],
            [9.1322657, 11.3909011],
            [9.1322275, 11.3908491],
            [9.1321646, 11.3908936],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320210", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324925, 11.3910954],
            [9.1325187, 11.3911088],
            [9.1325488, 11.3910523],
            [9.1325227, 11.3910389],
            [9.1324925, 11.3910954],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320211", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1324195, 11.3911427],
            [9.1324738, 11.3911644],
            [9.1324951, 11.3911131],
            [9.1324408, 11.3910914],
            [9.1324195, 11.3911427],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320212", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1323001, 11.3912177],
            [9.1323423, 11.3912426],
            [9.1323692, 11.391199],
            [9.1323269, 11.3911741],
            [9.1323001, 11.3912177],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320213", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1323779, 11.3912703],
            [9.1324362, 11.3913064],
            [9.1324612, 11.3912676],
            [9.1324029, 11.3912315],
            [9.1323779, 11.3912703],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320214", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1322216, 11.3912985],
            [9.1322518, 11.3913156],
            [9.1322673, 11.3912894],
            [9.1322371, 11.3912723],
            [9.1322216, 11.3912985],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320215", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1321278, 11.3912887],
            [9.132176, 11.3913143],
            [9.1322143, 11.391245],
            [9.132166, 11.3912194],
            [9.1321278, 11.3912887],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320221", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1335614, 11.3916167],
            [9.1335507, 11.3916679],
            [9.1336086, 11.3916796],
            [9.1336193, 11.3916283],
            [9.1335614, 11.3916167],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320222", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.133438, 11.3916088],
            [9.1333415, 11.3916022],
            [9.1333386, 11.3916431],
            [9.1334351, 11.3916497],
            [9.133438, 11.3916088],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320223", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1334715, 11.3915877],
            [9.1335332, 11.3915864],
            [9.1335321, 11.3915365],
            [9.1334704, 11.3915378],
            [9.1334715, 11.3915877],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320224", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1334394, 11.3913984],
            [9.1335104, 11.3914024],
            [9.1335125, 11.3913668],
            [9.1334414, 11.3913629],
            [9.1334394, 11.3913984],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320225", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1332717, 11.3913879],
            [9.133265, 11.3914576],
            [9.1333278, 11.3914634],
            [9.1333345, 11.3913937],
            [9.1332717, 11.3913879],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320226", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1331349, 11.3914431],
            [9.1331725, 11.391451],
            [9.133181, 11.3914122],
            [9.1331434, 11.3914043],
            [9.1331349, 11.3914431],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320227", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1328694, 11.3913261],
            [9.1328605, 11.3913682],
            [9.1329316, 11.3913827],
            [9.1329405, 11.3913406],
            [9.1328694, 11.3913261],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320228", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1329136, 11.391451],
            [9.132982, 11.3914681],
            [9.1329934, 11.3914243],
            [9.1329251, 11.3914072],
            [9.1329136, 11.391451],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320229", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1328037, 11.3914694],
            [9.1328451, 11.3914785],
            [9.1328666, 11.3913838],
            [9.1328251, 11.3913748],
            [9.1328037, 11.3914694],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320230", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1331738, 11.3916114],
            [9.1332219, 11.3916141],
            [9.133226, 11.3915444],
            [9.1331778, 11.3915417],
            [9.1331738, 11.3916114],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320231", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1330504, 11.3916403],
            [9.1331363, 11.3916482],
            [9.1331403, 11.3916061],
            [9.1330545, 11.3915982],
            [9.1330504, 11.3916403],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320232", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1327474, 11.3916482],
            [9.1327983, 11.3916732],
            [9.132819, 11.3916327],
            [9.132768, 11.3916077],
            [9.1327474, 11.3916482],
          ],
        ],
      },
    },
    {
      type: "Feature",
      property: { "@id": "way/734320233", building: "yes" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [9.1329029, 11.3916666],
            [9.1329432, 11.3916836],
            [9.1329714, 11.3916191],
            [9.1329311, 11.3916022],
            [9.1329029, 11.3916666],
          ],
        ],
      },
    },
  ],
};
// L.geoJSON(geojsonFeature2).addTo(mainMap);

var householdMarker = new L.Icon({
  iconUrl: "static/images/markers/marker-household.png",
  iconSize: [20, 20],
  iconAnchor: [10, 10],
  popupAnchor: [0, 0],
});

var hubMarker = new L.Icon({
  iconUrl: "/static/images/markers/marker-hub.png",
  iconSize: [20, 20],
  iconAnchor: [10, 10],
  popupAnchor: [0, 0],
});

var markers = [];
var lines = [];

mainMap.on("click", function (e) {
  var poplocation = e.latlng;

  if (mapClickEvent == "add_default_node") {
    addNodeToDatBase(poplocation.lat, poplocation.lng, "undefinded", false);
    drawDefaultMarker(poplocation.lat, poplocation.lng);
  }

  if (mapClickEvent == "add_fixed_household") {
    addNodeToDatBase(poplocation.lat, poplocation.lng, "household", true);
    drawHouseholdMarker(poplocation.lat, poplocation.lng);
  }

  if (mapClickEvent == "add_fixed_meterhub") {
    addNodeToDatBase(poplocation.lat, poplocation.lng, "meterhub", true);
    drawMeterhubMarker(poplocation.lat, poplocation.lng);
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

function getBuildingCoordinates(south, west, north, east) {
  var xhr = new XMLHttpRequest();
  // url = `https://www.overpass-api.de/api/interpreter?data=[out:json][timeout:2500][bbox:${south},${west},${north},${east}];(way["building"];relation["building"];);out body;>;out skel qt;`;
  url = "/validate_boundaries";
  console.log("URL:");
  console.log(url);
  xhr.open("POST", url, true);
  xhr.setRequestHeader("Content-Type", "application/json");

  // xhr.responseType = "text";
  console.log("sending request X");
  xhr.send(
    JSON.stringify({
      min_latitude: south,
      min_longitude: west,
      max_latitude: north,
      max_longitude: east,
    })
  );
  console.log("request sent");
  xhr.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      console.log("answer received");
      console.log(xhr.responseText);
      siteGeojson = JSON.parse(xhr.responseText);
      L.geoJSON(siteGeojson).addTo(mainMap);
    }
  };
}

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

function addNodeToDatBase(latitude, longitude, node_type, fixed_type) {
  $.ajax({
    url: "add_node/",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      latitude: latitude,
      longitude: longitude,
      node_type: node_type,
      fixed_type: fixed_type,
    }),
    dataType: "json",
  });
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

function optimize_grid(
  price_meterhub,
  price_household,
  price_interhub_cable,
  price_distribution_cable
) {
  $.ajax({
    url: "optimize_grid/",
    type: "POST",
    contentType: "application/json",
    data: JSON.stringify({
      price_meterhub: price_meterhub,
      price_household: price_household,
      price_interhub_cable: price_interhub_cable,
      price_distribution_cable: price_distribution_cable,
    }),
    dataType: "json",
  });
}

function refreshNodeTable() {
  var tbody_nodes = document.getElementById("tbody_nodes");
  var xhr = new XMLHttpRequest();
  url = "nodes_db_html";
  xhr.open("GET", url, true);
  xhr.responseType = "json";
  xhr.send();

  xhr.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      nodes = this.response;
      html_node_table = "";
      for (node of nodes) {
        html_node_table += `
              <tr>
              <td>${node.id}</td>
              <td>${node.latitude}</td>
              <td>${node.longitude}</td>
              <td>${node.node_type}</td>
              <td>${node.fixed_type}</td>
              </tr>`;
      }
      tbody_nodes.innerHTML = html_node_table;
      for (marker of markers) {
        mainMap.removeLayer(marker);
      }
      markers.length = 0;
      for (node of nodes) {
        if (node.node_type === "meterhub") {
          markers.push(
            L.marker([node.latitude, node.longitude], {
              icon: hubMarker,
            }).addTo(mainMap)
          );
        } else if (node.node_type === "household") {
          markers.push(
            L.marker([node.latitude, node.longitude], {
              icon: householdMarker,
            }).addTo(mainMap)
          );
        } else {
          markers.push(
            L.marker([node.latitude, node.longitude]).addTo(mainMap)
          );
        }
      }
    }
  };
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

function refreshLinkTable() {
  var tbody_links = document.getElementById("tbody_links");
  var xhr = new XMLHttpRequest();
  url = "links_db_html";
  xhr.open("GET", url, true);
  xhr.responseType = "json";
  xhr.send();

  xhr.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      links = this.response;
      html_link_table = "";
      for (link of links) {
        html_link_table += `
              <tr>
              <td>${link.id}</td>
              <td>${link.lat_from}</td>
              <td>${link.long_from}</td>
              <td>${link.lat_to}</td>
              <td>${link.long_to}</td>
              <td>${link.cable_type}</td>
              <td>${link.distance}</td>
              </tr>`;
      }
      tbody_links.innerHTML = html_link_table;
      ereaseLinksFromMap(mainMap);
      for (link of links) {
        var color = link.cable_type === "interhub" ? "red" : "green";
        drawLinkOnMap(
          link.lat_from,
          link.long_from,
          link.lat_to,
          link.long_to,
          color,
          mainMap
        );
      }
    }
  };
}

$(document).ready(function () {
  refreshNodeTable();
  refreshLinkTable();

  setInterval(refreshNodeTable, 3000);
  setInterval(refreshLinkTable, 3000);

  $("#button_add_undefined_node").click(function () {
    mapClickEvent = "add_default_node";
  });

  $("#button_add_household").click(function () {
    mapClickEvent = "add_fixed_household";
  });

  $("#button_add_meterhub").click(function () {
    mapClickEvent = "add_fixed_meterhub";
  });

  $("#button_add_node").click(function () {
    const latitude = new_node_lat.value;
    const longitude = new_node_long.value;
    const node_type = new_node_type.value;
    const fixed_type = new_node_type_fixed.value;

    addNodeToDatBase(latitude, longitude, node_type, fixed_type);
  });

  $("#button_optimize").click(function () {
    const price_hub = hub_price.value;
    const price_household = household_price.value;
    const price_interhub_cable = interhub_cable_price.value;
    const price_distribution_cable = distribution_cable_price.value;
    optimize_grid(
      price_hub,
      price_household,
      price_interhub_cable,
      price_distribution_cable
    );
  });

  $("#button_clear_node_db").click(function () {
    $.ajax({
      url: "clear_node_db/",
      type: "POST",
    });
  });

  $("#button_select_boundaries").click(function () {
    mapClickEvent = "draw_boundaries";
    var textSelectBoundaryButton = document.getElementById(
      "button_select_boundaries"
    );
    textSelectBoundaryButton.innerHTML = "Reset boundaries";
    removeBoundaries();
  });

  $("#button_validate_boundaries").click(function () {
    mapClickEvent = "select";

    // Close polygone by changing dashed line to solid
    if (dashedBoundaryLine != null) {
      mainMap.removeLayer(dashedBoundaryLine);
    }
    siteBoundaryLines.push(
      L.polyline([siteBoundaries[0], siteBoundaries.slice(-1)[0]], {
        color: "black",
      })
    );
    siteBoundaryLines[siteBoundaryLines.length - 1].addTo(mainMap);

    // Find most extreme latitudes and longitudes
    const latitudeList = siteBoundaries.map((x) => x[0]);
    const longitudeList = siteBoundaries.map((x) => x[1]);

    minLatitude = Math.min(...latitudeList);
    maxLatitude = Math.max(...latitudeList);

    minLongitude = Math.min(...longitudeList);
    maxLongitude = Math.max(...longitudeList);

    // TODO implement if close to check that area is not too large

    getBuildingCoordinates(
      (south = minLatitude),
      (west = minLongitude),
      (north = maxLatitude),
      (east = maxLongitude)
    );
  });
});
