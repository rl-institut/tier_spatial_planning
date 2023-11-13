
---
#  The Tool
The open-source tool originated from the PeopleSuN project and serves the planning of off-grid systems in Nigeria. The tool aims to perform a spatial optimization of the distribution grid as well as the design of the energy converters and energy storage.
## Features
The features of the tool are listed below:
- **Automatic Identification of Buildings from OpenStreetMap:** Utilizes OpenStreetMap data to automatically identify building locations.
- **Spatial Optimization of the Distribution Grid:** Enhances the efficiency of the distribution grid through spatial optimization techniques.
- **Design Optimization of Generation Systems:** Optimizes the design of PV systems, battery systems, inverters, and diesel-engines.
- **Automatic Identification for Individual Solar Home Systems:** Identifies buildings that are more suitably served by individual solar home systems.
---
# Instructions for Setting Up the Project
## Running the App in Docker Environment (Production Environment)
1. **Download the Project:**
   First, download the project from GitHub by visiting [this link](https://github.com/rl-institut/tier_spatial_planning/).
2. **Switch to the Project Directory:**
   After downloading, open your terminal and switch to the project directory by running:
   ```bash
   cd path/to/project_data
   ```
   Replace `path/to/project_data` with the actual path where you downloaded the project.
3. **Run Docker Compose on Linux:**
   If you're using a Linux system, set up your Docker environment by executing the following command in the terminal. This command sets your user ID and group ID for the database and starts the Docker containers:
   ```bash
   UID_FOR_DB=$(id -u) GID_FOR_DB=$(id -g) docker-compose up -d
   ```
## Running the App without Docker Environment (Development Environment):
1. **Install MySQL:** Begin by manually installing MySQL. If you prefer a graphical interface, you may also install MySQL Workbench.
2. **Create Database User:**
   - Create a new database user with the username specified in your `fastapi_app.env` file.
   - Set the password for this user as defined in your Docker `secret.txt`.
3. **Grant Privileges and Create Database:**
   - Grant the necessary privileges to the newly created user.
   - Create a database with the name you've specified in `fastapi_app.env`.
4. **Run the Application:**
   - Execute `run.py` to start the application. This script uses Uvicorn to run the app.
   - Once running, the web application should be accessible at [http://localhost:8080](http://localhost:8080).
   - When the app is executed for the first time, it automatically creates all necessary database tables and initiates the process of importing weather data into the database. This import process usually takes several minutes, potentially up to half an hour. During this time, please refrain from interrupting the process to ensure a complete and successful setup.
---
