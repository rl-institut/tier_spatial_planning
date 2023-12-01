from fastapi_app.python.db import sync_inserts, sync_queries

"""
This module is responsible for initializing the database upon the application's startup. It includes the init_db 
function which checks for the presence of weather data in the database. If weather data is absent and a CDS API key is 
available, the function triggers the download and update of weather data from the CDS. Additionally, the module ensures 
the creation of default database users 'admin' and 'default_example', if they do not already exist. This module is 
executed exclusively in a Dockerized environment by the Docker service 'db_initializer'. In a non-Dockerized 
environment, the same functions are executed by the application's startup event. This ensures that when Gunicorn runs 
multiple instances of the app, the database initialization is only performed once, rather than multiple times. Using a 
separate Docker service 'db_initializer', as opposed to manipulating the entrypoint of the app service, has the 
advantage of making the app container available and testable more quickly without waiting for the potentially lengthy 
(up to 2 hours) download process.
"""

def init_db():
    if not sync_queries.check_if_weather_data_exists():
        sync_inserts.update_weather_db()
    sync_inserts.create_default_user_account()


if __name__ == '__main__':
    init_db()