# Description

This open-source computer tool is part of the [PeopleSuN project](https://www.peoplesun.org), which contains a web-based app that aims to optimize the network structure as well as the power supply system for mini-grid systems.

The features of the tool are listed below:

- Automatic identification of buildings from the OpenStreetMap inside a given boundary.
- Network optimization of mini-grids based on the Python package [sgdot-lite](https://github.com/fsumpa/sgdot-lite).
- Optimization of the power supply systems for mini-grids (PV, battery, and diesel) using a Python-based tool called [Offgridders](https://github.com/rl-institut/offgridders). 
- Automatic identification of buildings that are better to be served by idividual solar home systems.


Install mysql (workbench)
Create user with name you fastapi_app.env and password in secret PW and grant privileges to the user to change db offgridplanner
create schema offgridplanner

