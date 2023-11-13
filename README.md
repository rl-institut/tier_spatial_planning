# Description

This open-source computer tool is part of the [PeopleSuN project](https://www.peoplesun.org), which contains a web-based app that aims to optimize the network structure as well as the power supply system for mini-grid systems.

The features of the tool are listed below:

- Automatic identification of buildings from the OpenStreetMap inside a given boundary.
- Grid optimization of mini-grids based on the Python package [sgdot-lite](https://github.com/fsumpa/sgdot-lite).
- Optimization of the power supply systems for mini-grids (PV, battery, and diesel) using a Python-based tool called [Offgridders](https://github.com/rl-institut/offgridders). 
- Automatic identification of buildings that are better to be served by idividual solar home systems.



# Running the App Without Docker Environment (Development Environment)
1. Install MySQL: Begin by manually installing MySQL. If you prefer a graphical interface, you may also install MySQL Workbench.
2. Create Database User:
- Create a new database user with the username specified in your fastapi_app.env file.
- Set the password for this user as defined in your Docker secret.txt.
3. Grant Privileges and Create Database:
- Grant the necessary privileges to the newly created user.
- Create a schema with the name you've specified in fastapi_app.env (DB_NAME).
4. Run the Application:
- Execute run.py to start the application. This script uses Uvicorn to run the app.
- Once running, the web application should be accessible at http://localhost:8080.
- When the app is executed for the first time, it automatically creates all necessary database tables and initiates the process of importing weather data into the database. This import process usually takes several minutes, potentially up to half an hour. During this time, please refrain from interrupting the process to ensure a complete and successful setup.